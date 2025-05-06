import random
import numpy as np
from collections import deque
import os
import csv
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import traci
import traci.constants as tc

# ------------------
# Hyperparameters
# ------------------
NUM_AGENTS = 21  # e.g., 100 intersections
all_junctions = [f"{letter}{num}" for letter in "ABCDE" for num in range(5)]
junctions = [i for i in all_junctions if i not in ["A0", "A4", "E0", "E4"]]
NUM_EPISODES = 1000
TIME_STEPS = 500  # Increased from 300 to give vehicles more time to complete routes

GAMMA = .99
BATCH_SIZE = 32
BETA = 0.5            # mixing weight between global and local rewards (baseline for β regularization)
LR_ACTOR = 1e-5       # Lowered from 9.70286e-05
LR_CRITIC = 1e-4      # Lowered from 0.0036386
MIN_LR_ACTOR = 1e-6   # Adjusted minimum learning rates
MIN_LR_CRITIC = 1e-5
REPLAY_BUFFER_SIZE = 200_000
TAU = 0.005           # Reduced from 0.01 for smoother updates

# Exploration parameters
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EPSILON_DECAY = 0.98

# Temperature for annealing
INITIAL_TEMPERATURE = 1.0
TARGET_TEMPERATURE = .1
ANNEALING_RATE = .001

MIN_GREEN_TIME = 5.0  # Minimum time a light stays green
MAX_GREEN_TIME = 30.0 # Maximum time a light stays green

# Dimensions for state and action
STATE_DIM = 4         # e.g., [waiting time, queue length, current phase, traffic flow rate]
ACTION_DIM = 2        # 0: North-South green, 1: East-West green

# Add network update frequency parameter
NETWORK_UPDATE_FREQUENCY = 4  # Update networks every 4 time steps

# ---------------
# Reward Scaling Hyperparameters (new)
# ---------------
LOCAL_REWARD_SCALE = 0.5  # Increased from 0.1 to give more weight to local rewards
GLOBAL_REWARD_SCALE = 0.5 / NUM_AGENTS  # Increased from 0.1/NUM_AGENTS to give more weight to global rewards
WAITING_TIME_PENALTY = 1.0  # Increased penalty for waiting time
EMERGENCY_BRAKING_PENALTY = 5.0  # Increased penalty for emergency braking
THROUGHPUT_REWARD = 500.0  # Base throughput reward
CONGESTION_PENALTY = 2.0  # New penalty for congestion
QUEUE_LENGTH_PENALTY = 0.5  # New penalty for queue length

BASELINE_WAIT = 100.0  # Reduced baseline to make rewards more sensitive
MAX_WAITING_TIME = 1000.0  # Cap waiting time at 1000 seconds
MAX_QUEUE_LENGTH = 50.0  # Maximum queue length for normalization

# Track previous state for relative improvements
prev_waiting_times = {}
prev_queue_lengths = {}
prev_throughput = 0

# At the top, after your other hyperparams:
# Max per‑step global reward ≈ THROUGHPUT_REWARD (completion bonus)
# Max per‑step local  reward ≈ BASELINE_WAIT + (max emergency pen) + (max smooth reward)
MAX_LOCAL_REWARD = BASELINE_WAIT + (MAX_GREEN_TIME * EMERGENCY_BRAKING_PENALTY) + (NUM_AGENTS * 1.0)  
MAX_GLOBAL_REWARD = THROUGHPUT_REWARD

# Worst‑case combined per step = beta*MAX_GLOBAL + (1-beta)*MAX_LOCAL 
MAX_COMBINED_STEP = max(BETA*MAX_GLOBAL_REWARD + (1-BETA)*MAX_LOCAL_REWARD,
                        (1-BETA)*MAX_GLOBAL_REWARD + BETA*MAX_LOCAL_REWARD)

# Over TIME_STEPS steps:
REWARD_SCALE = MAX_COMBINED_STEP * TIME_STEPS
print(f"REWARD_SCALE: {REWARD_SCALE}", flush=True)

# Add at the top with other global variables
queue_lengths = []  # Track queue lengths over time
vehicle_routes = {}  # Track routes for all vehicles
vehicle_start_times = {}  # Track when vehicles enter the simulation

# Add at the top with other hyperparameters
TYPICAL_ROUTE_LENGTH = 400.0  # Typical route length in meters (5 edges × 80m)

# ------------------
# Neural Network Models
# ------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        action_probs = torch.softmax(self.fc4(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, global_state_dim, global_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim + global_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

# Define a BetaNetwork to dynamically predict β from the global state.
class BetaNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BetaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)  # Output a single scalar

    def forward(self, state):
        # Use a sigmoid to ensure β is in (0,1)
        beta = torch.sigmoid(self.fc2(torch.relu(self.fc1(state))))
        return beta

# ------------------
# Replay Buffer
# ------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filename):
        """Save the replay buffer to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filename):
        """Load the replay buffer from a file."""
        with open(filename, 'rb') as f:
            self.buffer = deque(pickle.load(f), maxlen=self.buffer.maxlen)

# Add state normalization
class StateNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.count = 0
        self.momentum = 0.99  # Momentum for running statistics
        
    def update(self, state):
        """Update normalization statistics with a new (raw) state using momentum."""
        self.count += 1
        # Update mean with momentum
        self.mean = self.momentum * self.mean + (1 - self.momentum) * state
        # Update variance with momentum
        self.std = self.momentum * self.std + (1 - self.momentum) * (state - self.mean) ** 2
        
    def normalize(self, state):
        """Normalize a state using current statistics."""
        return (state - self.mean) / (np.sqrt(self.std) + 1e-8)
    
    def normalize_local(self, local_state, agent_idx):
        """Normalize a local state using the corresponding slice of global statistics."""
        start_idx = agent_idx * STATE_DIM
        end_idx = start_idx + STATE_DIM
        local_mean = self.mean[start_idx:end_idx]
        local_std = np.sqrt(self.std[start_idx:end_idx])
        return (local_state - local_mean) / (local_std + 1e-8)
    
    def save(self, filename):
        """Save normalization statistics to a file."""
        with open(filename, 'wb') as f:
            pickle.dump((self.mean, self.std, self.count), f)
    
    def load(self, filename):
        """Load normalization statistics from a file."""
        with open(filename, 'rb') as f:
            self.mean, self.std, self.count = pickle.load(f)


# Initialize state normalizer with raw global state dimension.
state_normalizer = StateNormalizer(STATE_DIM * NUM_AGENTS)

# Initialize replay buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# ------------------
# Initialize Agents: Actors and Critics (and their target networks)
# ------------------
actors = [Actor(STATE_DIM, ACTION_DIM) for _ in range(NUM_AGENTS)]
critics = [Critic(STATE_DIM * NUM_AGENTS, ACTION_DIM * NUM_AGENTS) for _ in range(NUM_AGENTS)]
target_actors = [Actor(STATE_DIM, ACTION_DIM) for _ in range(NUM_AGENTS)]
target_critics = [Critic(STATE_DIM * NUM_AGENTS, ACTION_DIM * NUM_AGENTS) for _ in range(NUM_AGENTS)]

for i in range(NUM_AGENTS):
    target_actors[i].load_state_dict(actors[i].state_dict())
    target_critics[i].load_state_dict(critics[i].state_dict())

actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in actors]
critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in critics]

# Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts
actor_schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2) for optimizer in actor_optimizers]
critic_schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2) for optimizer in critic_optimizers]

# Instantiate the beta network and its optimizer.
beta_network = BetaNetwork(STATE_DIM * NUM_AGENTS)
beta_optimizer = optim.Adam(beta_network.parameters(), lr=1e-4)  # Lowered learning rate from 1e-3 to 1e-4

# ------------------
# Utility Functions for SUMO via TraCI
# ------------------
def get_junctions_edge_ids(junction_id):
    """
    Get all edge IDs for a given junction.
    """
    return traci.junction.getIncomingEdges(junction_id)

def convert_edge_to_lane_id(edge_id):
    """
    Convert an edge ID to a lane ID.
    """
    lane_ids = traci.edge.getLaneNumber(edge_id)
    return [edge_id + "_" + str(i) for i in range(lane_ids)]

def get_local_state(junction_id):
    """
    Retrieve the local state for an intersection.
    Note: current_phase is now taken once per junction.
    """
    edge_ids = get_junctions_edge_ids(junction_id)
    lane_ids = [lane for edge_id in edge_ids for lane in convert_edge_to_lane_id(edge_id)]
    waiting_time = 0
    queue_length = 0
    # Get the phase once instead of summing over lanes.
    current_phase = traci.trafficlight.getPhase(junction_id)
    traffic_flow_rate = 0
    for lane_id in lane_ids:
        waiting_time += traci.lane.getWaitingTime(lane_id)
        queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
        traffic_flow_rate += traci.lane.getLastStepVehicleNumber(lane_id)
    
    return np.array([waiting_time, queue_length, current_phase, traffic_flow_rate], dtype=np.float32)

def get_global_state():
    """
    Aggregate local states from all agents.
    """
    global_state = []
    for i in range(NUM_AGENTS):
        state = get_local_state(junctions[i])
        global_state.extend(state)
    return np.array(global_state, dtype=np.float32)

def select_action(actor, state, agent_idx, epsilon=0.0):
    """
    Select an action for a given state using the actor network.
    Adds stochasticity through both epsilon-greedy exploration and softmax sampling.
    """
    if random.random() < epsilon:
        # Random action for exploration
        return random.randint(0, ACTION_DIM - 1)
    
    # Normalize the local state before feeding it to the actor
    state_norm = state_normalizer.normalize_local(state, agent_idx)
    state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)  # add batch dimension
    action_probs = actor(state_tensor).detach().numpy().squeeze(0)
    action = np.random.choice(ACTION_DIM, p=action_probs)
    return action

def set_traffic_light(intersection_id, action):
    """
    Set the traffic light phase via traci with proper transitions.
    The model controls both the phase and duration.
    """
    current_phase = traci.trafficlight.getPhase(intersection_id)
    current_state = traci.trafficlight.getRedYellowGreenState(intersection_id)
    
    # If we're in a yellow phase, don't change the light
    if 'y' in current_state:
        return
        
    # Get the time since the last phase change
    time_since_last_switch = traci.trafficlight.getPhaseDuration(intersection_id)
    
    # If we're in a green phase and haven't met minimum green time, don't change
    if ('G' in current_state or 'g' in current_state) and time_since_last_switch < MIN_GREEN_TIME:
        return
        
    # Set the appropriate phase based on action
    if action == 0:  # switch to North-South green
        if current_phase == 2:  # coming from East-West green
            # First set yellow
            traci.trafficlight.setPhase(intersection_id, 3)
            traci.trafficlight.setPhaseDuration(intersection_id, 3)
        else:
            traci.trafficlight.setPhase(intersection_id, 0)
            # Set a long duration that the model can control
            traci.trafficlight.setPhaseDuration(intersection_id, MAX_GREEN_TIME)
    else:  # action == 1, switch to East-West green
        if current_phase == 0:  # coming from North-South green
            # First set yellow
            traci.trafficlight.setPhase(intersection_id, 1)
            traci.trafficlight.setPhaseDuration(intersection_id, 3)
        else:
            traci.trafficlight.setPhase(intersection_id, 2)
            # Set a long duration that the model can control
            traci.trafficlight.setPhaseDuration(intersection_id, MAX_GREEN_TIME)

# Add emergency braking tracking
emergency_braking_events = {}  # Track emergency braking events per junction
total_emergency_braking_events = {}  # Track total emergency braking events per junction

def track_emergency_braking():
    """
    Track emergency braking events for all vehicles in the network.
    """
    global emergency_braking_events, total_emergency_braking_events
    
    # Initialize total counts if not already done
    if not total_emergency_braking_events:
        total_emergency_braking_events = {junction: 0 for junction in junctions}
    
    # Reset emergency braking events for this time step
    emergency_braking_events = {junction: 0 for junction in junctions}
    
    # Get all vehicles in the network
    for veh_id in traci.vehicle.getIDList():
        # Get vehicle's current lane and junction
        lane_id = traci.vehicle.getLaneID(veh_id)
        if not lane_id:  # Skip if vehicle is not on a lane
            continue
            
        # Get the junction this lane belongs to
        edge_id = lane_id.split('_')[0]  # Get the edge ID from lane ID
        junction_id = None
        for j in junctions:
            if edge_id in get_junctions_edge_ids(j):
                junction_id = j
                break
                
        if junction_id:
            # Check for emergency braking
            decel = traci.vehicle.getDecel(veh_id)
            if decel > 4.5:  # Emergency braking threshold
                emergency_braking_events[junction_id] += 1
                total_emergency_braking_events[junction_id] += 1

def compute_reward(junction_id):
    """
    Compute the local reward for an intersection with relative improvements and normalized penalties.
    """
    edge_ids = get_junctions_edge_ids(junction_id)
    lane_ids = [lane for edge_id in edge_ids for lane in convert_edge_to_lane_id(edge_id)]
    
    # Base reward components
    waiting_time = 0
    emergency_braking_penalty = 0
    smooth_traffic_reward = 0
    queue_length = 0
    
    for lane_id in lane_ids:
        # Waiting time penalty (capped at MAX_WAITING_TIME)
        lane_waiting_time = traci.lane.getWaitingTime(lane_id)
        waiting_time += min(lane_waiting_time, MAX_WAITING_TIME)
        
        # Queue length penalty
        lane_queue = traci.lane.getLastStepHaltingNumber(lane_id)
        queue_length += lane_queue
        
        # Smooth traffic reward (reward for vehicles moving at desired speed)
        mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
        max_speed = traci.lane.getMaxSpeed(lane_id)
        if mean_speed > 0:
            speed_ratio = mean_speed / max_speed
            smooth_traffic_reward += speed_ratio
    
    # Get emergency braking events for this junction
    emergency_braking_penalty = emergency_braking_events.get(junction_id, 0)
    
    # Calculate relative improvements
    prev_wait = prev_waiting_times.get(junction_id, waiting_time)
    prev_queue = prev_queue_lengths.get(junction_id, queue_length)
    
    # Update previous values
    prev_waiting_times[junction_id] = waiting_time
    prev_queue_lengths[junction_id] = queue_length
    
    # Calculate relative improvements (negative means improvement)
    waiting_improvement = (waiting_time - prev_wait) / (prev_wait + 1e-6)
    queue_improvement = (queue_length - prev_queue) / (prev_queue + 1e-6)
    
    # Normalize rewards and penalties
    normalized_waiting = waiting_time / MAX_WAITING_TIME
    normalized_queue = queue_length / MAX_QUEUE_LENGTH
    
    # Combine rewards with relative improvements
    waiting_penalty = -WAITING_TIME_PENALTY * (normalized_waiting + waiting_improvement)
    queue_penalty = -QUEUE_LENGTH_PENALTY * (normalized_queue + queue_improvement)
    emergency_penalty = -EMERGENCY_BRAKING_PENALTY * emergency_braking_penalty
    smooth_reward = smooth_traffic_reward * 0.5
    
    # Add congestion penalty based on queue length
    congestion_penalty = -CONGESTION_PENALTY * (queue_length / MAX_QUEUE_LENGTH)
    
    # Combine all components
    return waiting_penalty + queue_penalty + emergency_penalty + smooth_reward + congestion_penalty

def compute_global_reward():
    """
    Compute the global reward with normalized travel times and route-length weighted completion.
    """
    global prev_throughput, vehicle_routes, vehicle_start_times
    
    # Track vehicle travel times and route lengths
    total_normalized_travel_time = 0.0
    total_route_length = 0.0
    completed_vehicles = 0
    
    # Update routes for new vehicles
    for veh_id in traci.vehicle.getIDList():
        if veh_id not in vehicle_routes:
            # New vehicle, store its route
            vehicle_routes[veh_id] = traci.vehicle.getRoute(veh_id)
            vehicle_start_times[veh_id] = traci.simulation.getTime()
    
    # Process arrived vehicles
    for veh_id in traci.simulation.getArrivedIDList():
        if veh_id in vehicle_routes:  # Only process if we have the route info
            # Calculate route length using lane lengths
            route_length = 0.0
            for edge_id in vehicle_routes[veh_id]:
                # Get all lanes for this edge and use the first lane's length
                # (in our grid, all lanes in an edge should have same length)
                lane_ids = traci.edge.getLaneNumber(edge_id)
                if lane_ids > 0:
                    lane_id = f"{edge_id}_0"  # First lane of the edge
                    route_length += traci.lane.getLength(lane_id)
            
            # Calculate travel time
            travel_time = traci.simulation.getTime() - vehicle_start_times[veh_id]
            
            if route_length > 0:  # Avoid division by zero
                normalized_travel_time = travel_time / route_length
                total_normalized_travel_time += normalized_travel_time
                total_route_length += route_length
                completed_vehicles += 1
            
            # Clean up stored data
            del vehicle_routes[veh_id]
            del vehicle_start_times[veh_id]
    
    # Calculate average normalized travel time
    if completed_vehicles > 0:
        avg_normalized_travel_time = total_normalized_travel_time / completed_vehicles
        avg_route_length = total_route_length / completed_vehicles
    else:
        avg_normalized_travel_time = 0
        avg_route_length = 0
    
    # Calculate route-length weighted completion rate
    if total_spawned_vehicles > 0:
        base_completion_rate = completed_vehicles / total_spawned_vehicles
        # Weight by average route length to favor longer routes
        weighted_completion = base_completion_rate * (avg_route_length / TYPICAL_ROUTE_LENGTH)  # Normalize by typical route length
    else:
        weighted_completion = 0
    
    # Calculate travel time improvement
    current_travel_time = avg_normalized_travel_time
    travel_time_improvement = (prev_throughput - current_travel_time) / (prev_throughput + 1e-6)
    prev_throughput = current_travel_time
    
    # Calculate completion reward with improvements
    completion_reward = weighted_completion * THROUGHPUT_REWARD * (1 + travel_time_improvement)
    
    # Calculate total waiting time penalty
    total_waiting_time = 0.0
    total_queue_length = 0.0
    for i in range(NUM_AGENTS):
        edge_ids = get_junctions_edge_ids(junctions[i])
        lane_ids = [lane for edge_id in edge_ids for lane in convert_edge_to_lane_id(edge_id)]
        for lane_id in lane_ids:
            total_waiting_time += traci.lane.getWaitingTime(lane_id)
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
    
    # Normalize global penalties
    normalized_waiting = total_waiting_time / (MAX_WAITING_TIME * NUM_AGENTS)
    normalized_queue = total_queue_length / (MAX_QUEUE_LENGTH * NUM_AGENTS)
    
    # Combine completion reward with normalized penalties
    return completion_reward - (normalized_waiting * WAITING_TIME_PENALTY) - (normalized_queue * QUEUE_LENGTH_PENALTY)

# ------------------
# Dynamic Reward Function with Scaling (Modified)
# ------------------
def compute_dynamic_reward(global_state, r_local, r_global, baseline_beta=0.5, lambda_reg=0.1):  # Increased lambda_reg from 0.01 to 0.1
    """
    Computes the mixed reward and returns the scalar reward plus beta regularization loss.
    BetaNetwork now receives gradients from combined reward.
    """
    # Scale the rewards
    r_local_scaled = LOCAL_REWARD_SCALE * r_local
    r_global_scaled = GLOBAL_REWARD_SCALE * r_global

    # Convert state to tensor
    state_tensor = torch.FloatTensor(global_state).unsqueeze(0)  # shape [1, global_dim]

    # Predict beta (between 0 and 1)
    beta_tensor = beta_network(state_tensor)  # shape [1,1]

    # Compute combined reward and regularization
    combined = beta_tensor * r_global_scaled + (1 - beta_tensor) * r_local_scaled
    reward = combined / REWARD_SCALE

    # Beta loss now includes negative reward term to maximize combined
    beta_reward_loss = -combined.mean()  # maximize combined scaled reward
    beta_reg_loss = lambda_reg * (beta_tensor - baseline_beta).pow(2).mean()
    beta_loss = beta_reward_loss + beta_reg_loss

    return reward.item(), beta_loss

# ------------------
# Soft Update Function for Target Networks
# ------------------
def soft_update(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# ------------------
# Update Networks (Critic and Actor)
# ------------------
def update_networks():
    """
    Update the actor and critic networks using experience replay.
    Returns the average actor and critic losses across all agents.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0, 0.0  # Return zero losses if not enough data
    
    # Sample a mini-batch of transitions
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states = torch.FloatTensor(states)  # shape: [BATCH_SIZE, global_state_dim]
    rewards = torch.FloatTensor(rewards)  # shape: [BATCH_SIZE, NUM_AGENTS]
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)
    
    # Convert actions to one-hot encoding:
    actions = torch.LongTensor(actions)  # shape: [BATCH_SIZE, NUM_AGENTS]
    actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=ACTION_DIM).float()
    actions_one_hot = actions_one_hot.view(BATCH_SIZE, -1)
    
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    
    for i in range(NUM_AGENTS):
        # ------------------
        # Critic Update: Minimize Bellman Error
        # ------------------
        next_actions = []
        for j in range(NUM_AGENTS):
            # Extract j-th agent's next state
            agent_next_state = next_states[:, j * STATE_DIM:(j + 1) * STATE_DIM]
            next_action_prob = target_actors[j](agent_next_state)
            # Taking argmax is acceptable for target estimation in discrete actions
            next_action = torch.argmax(next_action_prob, dim=1, keepdim=True)
            next_action_one_hot = torch.nn.functional.one_hot(next_action.squeeze(1), num_classes=ACTION_DIM).float()
            next_actions.append(next_action_one_hot)
        joint_next_actions = torch.cat(next_actions, dim=1)
        
        target_q = target_critics[i](next_states, joint_next_actions)
        y = rewards[:, i].unsqueeze(1) + GAMMA * target_q
        
        current_q = critics[i](states, actions_one_hot)
        critic_loss = nn.MSELoss()(current_q, y.detach())
        
        critic_optimizers[i].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critics[i].parameters(), max_norm=1.0)
        critic_optimizers[i].step()
        
        # ------------------
        # Actor Update: Deterministic Policy Gradient (MADDPG-style)
        # ------------------
        # Get local (agent i) state from the global batch
        agent_state = states[:, i * STATE_DIM:(i + 1) * STATE_DIM]
        
        # Get the best action according to current policy
        current_action_prob = actors[i](agent_state)
        best_action = torch.argmax(current_action_prob, dim=1, keepdim=True)
        best_one_hot = torch.nn.functional.one_hot(best_action.squeeze(1), num_classes=ACTION_DIM).float()
        
        # Construct joint action with best action for agent i
        joint_best = actions_one_hot.clone().view(BATCH_SIZE, NUM_AGENTS, ACTION_DIM)
        joint_best[:, i, :] = best_one_hot
        joint_best = joint_best.view(BATCH_SIZE, -1)
        
        # Actor loss is negative Q-value of the best action
        actor_loss = -critics[i](states, joint_best).mean()
        
        # Update actor parameters
        actor_optimizers[i].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actors[i].parameters(), max_norm=1.0)
        actor_optimizers[i].step()
        
        # Accumulate losses
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        
        # ------------------
        # Soft Update Target Networks
        # ------------------
        soft_update(actors[i], target_actors[i], TAU)
        soft_update(critics[i], target_critics[i], TAU)
    
    # Return average losses across all agents
    return total_actor_loss / NUM_AGENTS, total_critic_loss / NUM_AGENTS

# ------------------
# Functions for Logging and Saving Metrics/Models
# ------------------
def get_simulation_metrics(actor_loss=None, critic_loss=None, beta_value=None, lr_actor=None, lr_critic=None):
    """
    Get additional simulation metrics including training metrics.
    """
    
    metrics = {
        'vehicles_in_network': total_spawned_vehicles,
        'arrived_vehicles': total_arrived_vehicles,
        'average_speed': 0.0,
        'average_travel_time': 0.0,
        'emergency_braking_events': sum(emergency_braking_events.values()),  # Current time step events
        'total_emergency_braking_events': sum(total_emergency_braking_events.values()),  # Total events
        'queue_length': queue_lengths[-1] if queue_lengths else 0,  # Use the last tracked queue length
        'average_queue_length': sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0,  # Use tracked queue lengths
        'actor_loss': actor_loss if actor_loss is not None else 0.0,
        'critic_loss': critic_loss if critic_loss is not None else 0.0,
        'beta_value': beta_value if beta_value is not None else 0.0,
        'lr_actor': lr_actor if lr_actor is not None else 0.0,
        'lr_critic': lr_critic if lr_critic is not None else 0.0
    }
    
    # Calculate average speed and travel time
    total_speed = 0
    total_travel_time = 0
    vehicle_count = 0
    
    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)
        if speed > 0:  # Only count moving vehicles
            total_speed += speed
            # Get travel time using getTimeLoss instead of getTravelTime
            total_travel_time += traci.vehicle.getTimeLoss(veh_id)
            vehicle_count += 1
    
    if vehicle_count > 0:
        metrics['average_speed'] = total_speed / vehicle_count
        metrics['average_travel_time'] = total_travel_time / vehicle_count
    
    # Get total waiting time
    total_waiting_time = 0
    for veh_id in traci.vehicle.getIDList():
        total_waiting_time += traci.vehicle.getTimeLoss(veh_id)
    
    # Get average waiting time
    metrics['average_waiting_time'] = total_waiting_time / total_spawned_vehicles if total_spawned_vehicles > 0 else 0.0
    
    return metrics

def init_metrics_file(trial_dir=None):
    """
    Initialize the metrics file and configuration file in the Ray Tune trial directory.
    If no trial_dir is provided, create a local directory for testing.
    """
    run_dir = "/app/src/runs"
    # Create metrics file with more detailed columns
    metrics_file = os.path.join(run_dir, "metrics.csv")
    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Episode", 
            "Cumulative Reward", 
            "Average Waiting Time", 
            "Vehicles in Network",
            "Arrived Vehicles",
            "Average Speed",
            "Average Travel Time",
            "Total Emergency Braking Events",
            "Queue Length",
            "Average Queue Length",
            "Actor Loss",
            "Critic Loss",
            "Beta Value",
            "Learning Rate Actor",
            "Learning Rate Critic"
        ])
    
    return metrics_file

def log_metrics(metrics_file, episode, episode_reward, total_arrived_vehicles, total_spawned_vehicles,
                sim_metrics, actor_loss, critic_loss, beta_value, lr_actor, lr_critic):
    """
    Log detailed metrics to the specified file.
    """
    with open(metrics_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            episode, 
            episode_reward, 
            sim_metrics['average_waiting_time'], 
            total_arrived_vehicles,
            total_spawned_vehicles,
            sim_metrics['average_speed'],
            sim_metrics['average_travel_time'],
            sim_metrics['total_emergency_braking_events'],
            sim_metrics['queue_length'],
            sim_metrics['average_queue_length'],
            actor_loss,
            critic_loss,
            beta_value,
            lr_actor,
            lr_critic
        ])

def save_model_checkpoint(episode):
    """
    Save the current model checkpoint and delete older checkpoints.
    Only keeps the latest checkpoint to save disk space.
    """
    # Ensure the models directory exists.
    os.makedirs("/app/src/models", exist_ok=True)
    
    # Delete old checkpoint files.
    for filename in os.listdir("/app/src/models"):
        if filename.endswith(".pth") or filename.endswith(".pkl"):
            os.remove(os.path.join("/app/src/models", filename))
    
    # Save new checkpoint.
    for i in range(NUM_AGENTS):
        torch.save(actors[i].state_dict(), f"/app/src/models/actor_agent_{i}_episode_{episode}.pth")
        torch.save(critics[i].state_dict(), f"/app/src/models/critic_agent_{i}_episode_{episode}.pth")
    
    # Save replay buffer and state normalizer.
    replay_buffer.save(f"/app/src/models/replay_buffer_episode_{episode}.pkl")
    state_normalizer.save(f"/app/src/models/state_normalizer_episode_{episode}.pkl")
    
    print(f"Checkpoint saved for episode {episode} (deleted older checkpoints)")

def load_model_checkpoint(episode):
    """
    Load the model checkpoint from the specified episode and reset learning rates.
    """
    for i in range(NUM_AGENTS):
        actors[i].load_state_dict(torch.load(f"/app/src/models/actor_agent_{i}_episode_{episode}.pth"))
        critics[i].load_state_dict(torch.load(f"/app/src/models/critic_agent_{i}_episode_{episode}.pth"))
    
    # Load replay buffer and state normalizer.
    replay_buffer.load(f"/app/src/models/replay_buffer_episode_{episode}.pkl")
    state_normalizer.load(f"/app/src/models/state_normalizer_episode_{episode}.pkl")
    
    # Reset learning rates for all optimizers
    for optimizer in actor_optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR_ACTOR  # Reset to initial value
    
    for optimizer in critic_optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR_CRITIC  # Reset to initial value
    
    print(f"Checkpoint loaded from episode {episode} and learning rates reset to initial values")

# Add a function to get and log learning rates.
def log_learning_rates(episode):
    """Log the current learning rates for all networks."""
    actor_lrs = [scheduler._last_lr[0] for scheduler in actor_schedulers]
    critic_lrs = [scheduler._last_lr[0] for scheduler in critic_schedulers]
    # Log to file instead of printing
    with open('src/logs/learning_rates.log', 'a') as f:
        f.write(f"Episode {episode} - Actor LRs: {actor_lrs}, Critic LRs: {critic_lrs}\n")

# Add per-agent reward tracking
per_agent_rewards = [[] for _ in range(NUM_AGENTS)]

def compute_reward_variance(per_agent_rewards):
    """Compute the variance of rewards across agents."""
    if not any(per_agent_rewards):  # If no rewards recorded yet
        return 0.0
    # Get the mean reward for each agent
    agent_means = [np.mean(rewards) if rewards else 0.0 for rewards in per_agent_rewards]
    # Compute variance across agent means
    return np.var(agent_means)


def generate_random_trips():
    """
    Generate new random trips by modifying the existing trips file.
    This is a simpler approach that doesn't require SUMO's Python tools.
    """
    # Read the existing trips file.
    with open('src/net/trips.trips.xml', 'r') as f:
        trips_content = f.read()
    
    # Find all trip elements.
    import re
    trips = re.findall(r'<trip.*?/>', trips_content)
    
    # Randomly shuffle the trips.
    import random
    random.shuffle(trips)
    
    # Take 500 trips for one episode.
    trips_per_episode = 1500
    selected_trips = trips[:trips_per_episode]
    
    # Create new trips with random depart times.
    new_trips = []
    for i, trip in enumerate(selected_trips):
        # Extract the original trip attributes.
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', trip))
        
        # Generate a random depart time between 0 and TIME_STEPS.
        depart = random.uniform(0, TIME_STEPS)
        
        # Create new trip with random depart time.
        new_trip = (depart, f'<trip id="{i}" depart="{depart:.2f}" from="{attrs["from"]}" to="{attrs["to"]}"/>')
        new_trips.append(new_trip)
    
    # Sort trips by departure time.
    new_trips.sort(key=lambda x: x[0])
    
    # Write the new trips file with a different name.
    with open('src/net/random_trips.trips.xml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('\n'.join(trip[1] for trip in new_trips))
        f.write('\n</routes>')
    
    # Convert trips to routes using duarouter with the new file.
    import os
    os.system('duarouter -n src/net/grid5x5.net.xml -r src/net/random_trips.trips.xml -o src/net/grid5x5.rou.xml --ignore-errors')

# ------------------
# Main Training Loop
# ------------------
def run_training():
    global total_arrived_vehicles
    global total_spawned_vehicles
    global queue_lengths  # Add this line
    
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Run traffic simulation training')
    parser.add_argument('--load-checkpoint', type=int, help='Episode number of checkpoint to load')
    args = parser.parse_args()

    # Start SUMO with traci.
    sumoBinary = "sumo"  # or "sumo-gui" for visualization.
    sumoConfig = "src/net/grid5x5.sumocfg"
    sumoCommand = [sumoBinary, "-c", sumoConfig]

    # Initialize metrics file and get run directory
    metrics_file = init_metrics_file()
    
    # Handle checkpoint loading from config instead of command line
    start_episode = 0

    # Track average reward per episode
    total_episodes = 0
    total_reward = 0
    
    # Early stopping parameters
    best_reward = float('-inf')
    patience_counter = 0
    early_stop = False
    
    epsilon = INITIAL_EPSILON

    # Initialize per-agent reward tracking
    per_agent_rewards = [[] for _ in range(NUM_AGENTS)]
    
    # Run training starting from the specified episode
    for episode in range(start_episode, NUM_EPISODES):
        if early_stop:
            print(f"Early stopping triggered at episode {episode}")
            break
            
        print(f"Episode {episode+1}")
        
        # Reset queue lengths for new episode
        queue_lengths = []
        
        # Get random routes for the vehicles
        generate_random_trips()
        
        # Start SUMO.
        sumoBinary = "sumo"  # or "sumo-gui" for visualization.
        sumoConfig = "/app/src/net/grid5x5.sumocfg"
        sumoCommand = [sumoBinary, "-c", sumoConfig]
        traci.start(sumoCommand)
        
        # Get the raw global state then normalize and update the state normalizer (using raw values).
        raw_state = get_global_state()
        global_state = state_normalizer.normalize(raw_state)
        state_normalizer.update(raw_state)
        
        episode_reward = 0
        
        total_arrived_vehicles = 0
        total_spawned_vehicles = 0
        
        # Initialize losses to ensure they're always defined
        actor_loss = critic_loss = 0.0
        
        for t in range(TIME_STEPS):
            # Track emergency braking events at the start of each time step
            track_emergency_braking()
            
            # Calculate and store current queue length
            current_queue_length = 0
            for lane_id in traci.lane.getIDList():
                current_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
            queue_lengths.append(current_queue_length)
            
            # set done to true on last time step
            done = t == TIME_STEPS - 1
            actions = []
            for i in range(NUM_AGENTS):
                junction_id = junctions[i]
                local_state = get_local_state(junction_id)
                action = select_action(actors[i], local_state, i, epsilon)
                actions.append(action)
                set_traffic_light(junction_id, action)
            
            # Advance simulation one step.
            traci.simulationStep()
            
            # Update global metrics
            total_arrived_vehicles += traci.simulation.getArrivedNumber()
            total_spawned_vehicles += traci.simulation.getDepartedNumber()
            
            # Get next state using raw state values.
            raw_next_state = get_global_state()
            next_global_state = state_normalizer.normalize(raw_next_state)
            state_normalizer.update(raw_next_state)
            
            # Calculate rewards
            local_rewards = []
            total_beta_loss = None
            
            # Compute global reward once per time step
            r_global_step = compute_global_reward()
            
            for i in range(NUM_AGENTS):
                r_local = compute_reward(junctions[i])
                r, beta_loss = compute_dynamic_reward(global_state, r_local, r_global_step)
                if total_beta_loss is None:
                    total_beta_loss = beta_loss
                else:
                    total_beta_loss += beta_loss
                local_rewards.append(r)
                episode_reward += r
                # Track per-agent rewards
                per_agent_rewards[i].append(r)
            
            # Apply beta loss to beta network
            beta_optimizer.zero_grad()
            total_beta_loss.backward()
            torch.nn.utils.clip_grad_norm_(beta_network.parameters(), max_norm=1.0)  # Added gradient clipping
            beta_optimizer.step()
            
            # Store transition in the replay buffer.
            replay_buffer.push(global_state, actions, local_rewards, next_global_state, done)
            global_state = next_global_state
            
            # Update networks periodically
            if t % NETWORK_UPDATE_FREQUENCY == 0:
                actor_loss, critic_loss = update_networks()
            
            # Ensure learning rates don't go below minimum
            for i in range(NUM_AGENTS):
                for param_group in actor_optimizers[i].param_groups:
                    param_group['lr'] = max(param_group['lr'], MIN_LR_ACTOR)
                for param_group in critic_optimizers[i].param_groups:
                    param_group['lr'] = max(param_group['lr'], MIN_LR_CRITIC)
        
        # Step the schedulers at the end of each episode
        for i in range(NUM_AGENTS):
            actor_schedulers[i].step(episode)  # Pass the episode number as the step
            critic_schedulers[i].step(episode)  # Pass the episode number as the step
        
        # Decay epsilon
        epsilon = max(FINAL_EPSILON, epsilon * EPSILON_DECAY)
        
        # Get simulation metrics with training metrics
        sim_metrics = get_simulation_metrics(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            beta_value=beta_network(torch.FloatTensor(global_state)).mean().item(),
            lr_actor=actor_optimizers[0].param_groups[0]['lr'],
            lr_critic=critic_optimizers[0].param_groups[0]['lr']
        )
        
        # Log metrics
        log_metrics(
            metrics_file, episode+1, episode_reward, total_arrived_vehicles, total_spawned_vehicles,
            sim_metrics, actor_loss, critic_loss, beta_network(torch.FloatTensor(global_state)).mean().item(),
            actor_optimizers[0].param_groups[0]['lr'],
            critic_optimizers[0].param_groups[0]['lr']
        )
        
        traci.close()  # End the simulation for this episode.
        
        # Save checkpoint every 50 episodes.
        if (episode + 1) % 50 == 0:
            save_model_checkpoint(episode + 1)
            
        # Update total reward and episode count
        total_reward += episode_reward
        total_episodes += 1
        
        # Compute reward variance at the end of each episode
        reward_variance = compute_reward_variance(per_agent_rewards)
        
        print(f"Episode {episode+1} - Reward Variance: {reward_variance}", flush=True)
        print(f"Episode {episode+1} - Reward: {episode_reward}", flush=True)
        
        # Clear per-agent rewards for next episode
        per_agent_rewards = [[] for _ in range(NUM_AGENTS)]
        
    # Save final models.
    save_model_checkpoint("final")

if __name__ == '__main__':
    run_training()
