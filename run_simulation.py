import os
import sys
import numpy as np
import torch
import traci
import traci.constants as tc
from collections import deque
import pickle
import json
import time
import random

# Import necessary components from main2.py
from main import (
    Actor, Critic, StateNormalizer, ReplayBuffer,
    get_local_state, get_global_state, set_traffic_light,
    STATE_DIM, ACTION_DIM, NUM_AGENTS, junctions,
    MIN_GREEN_TIME, MAX_GREEN_TIME
)

def load_models(model_dir, episode):
    """Load the trained models and state normalizer."""
    # Load state normalizer data
    with open(os.path.join(model_dir, f"state_normalizer_episode_{episode}.pkl"), 'rb') as f:
        mean, std, count = pickle.load(f)
    
    # Create and initialize state normalizer
    state_normalizer = StateNormalizer(STATE_DIM * NUM_AGENTS)
    state_normalizer.mean = mean
    state_normalizer.std = std
    state_normalizer.count = count
    
    # Initialize and load actor networks
    actors = []
    for i in range(NUM_AGENTS):
        actor = Actor(STATE_DIM, ACTION_DIM)
        actor.load_state_dict(torch.load(os.path.join(model_dir, f"actor_agent_{i}_episode_{episode}.pth")))
        actor.eval()  # Set to evaluation mode
        actors.append(actor)
    
    return actors, state_normalizer

def select_action(actor, state, state_normalizer, agent_idx, epsilon=0.1):
    """Select an action using the trained actor network with epsilon-greedy exploration."""
    if random.random() < epsilon:
        # Random action for exploration
        return random.randint(0, ACTION_DIM - 1)
    
    # Normalize the state
    state_norm = state_normalizer.normalize_local(state, agent_idx)
    # Clip normalized state to reasonable range
    state_norm = np.clip(state_norm, -3, 3)
    state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
    
    # Get action probabilities
    with torch.no_grad():
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()
    
    return action

def set_traffic_light_with_safety(intersection_id, action):
    """
    Set the traffic light phase with safety checks and proper transitions.
    """
    current_phase = traci.trafficlight.getPhase(intersection_id)
    current_state = traci.trafficlight.getRedYellowGreenState(intersection_id)
    time_since_last_switch = traci.trafficlight.getPhaseDuration(intersection_id)
    
    # If we're in a yellow phase, don't change the light
    if 'y' in current_state:
        return
        
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
            traci.trafficlight.setPhaseDuration(intersection_id, MAX_GREEN_TIME)
    else:  # action == 1, switch to East-West green
        if current_phase == 0:  # coming from North-South green
            # First set yellow
            traci.trafficlight.setPhase(intersection_id, 1)
            traci.trafficlight.setPhaseDuration(intersection_id, 3)
        else:
            traci.trafficlight.setPhase(intersection_id, 2)
            traci.trafficlight.setPhaseDuration(intersection_id, MAX_GREEN_TIME)

def run_simulation(model_dir, episode, num_steps=1000, gui=True):
    """Run a SUMO simulation using the trained models."""
    # Load models
    actors, state_normalizer = load_models(model_dir, episode)
    
    # Start SUMO
    if gui:
        sumoBinary = "sumo-gui"
    else:
        sumoBinary = "sumo"
    
    sumoConfig = "/app/src/net/grid5x5.sumocfg"
    sumoCommand = [sumoBinary, "-c", sumoConfig]
    traci.start(sumoCommand)
    
    # Initialize metrics
    total_waiting_time = 0
    total_vehicles = 0
    total_arrived = 0
    
    # Debug: Track action distribution
    action_counts = {0: 0, 1: 0}
    
    try:
        # Run simulation
        for step in range(num_steps):
            # Get current state
            global_state = get_global_state()
            
            # Select and execute actions for each agent
            for i in range(NUM_AGENTS):
                junction_id = junctions[i]
                local_state = get_local_state(junction_id)
                action = select_action(actors[i], local_state, state_normalizer, i, epsilon=0.1)
                action_counts[action] += 1
                set_traffic_light_with_safety(junction_id, action)
                
                # Debug: Print state and action for first junction
                if i == 0 and step % 100 == 0:
                    print(f"\nJunction {junction_id} at step {step}:")
                    print(f"Raw state: {local_state}")
                    print(f"Normalized state: {state_normalizer.normalize_local(local_state, i)}")
                    print(f"Selected action: {action}")
                    print(f"Current phase: {traci.trafficlight.getPhase(junction_id)}")
                    print(f"Current state: {traci.trafficlight.getRedYellowGreenState(junction_id)}")
            
            # Advance simulation
            traci.simulationStep()
            
            # Update metrics
            total_arrived += traci.simulation.getArrivedNumber()
            total_vehicles += traci.simulation.getDepartedNumber()
            
            # Calculate waiting time
            for i in range(NUM_AGENTS):
                edge_ids = traci.junction.getIncomingEdges(junctions[i])
                for edge_id in edge_ids:
                    lane_ids = [edge_id + "_" + str(i) for i in range(traci.edge.getLaneNumber(edge_id))]
                    for lane_id in lane_ids:
                        total_waiting_time += traci.lane.getWaitingTime(lane_id)
            
            # Print progress
            if step % 100 == 0:
                print(f"\nStep {step}/{num_steps}")
                print(f"Vehicles arrived: {total_arrived}")
                print(f"Average waiting time: {total_waiting_time / (step + 1):.2f}")
                print(f"Action distribution: {action_counts}")
                print("---")
    
    finally:
        traci.close()
    
    # Print final metrics
    print("\nSimulation Complete!")
    print(f"Total vehicles: {total_vehicles}")
    print(f"Total arrived: {total_arrived}")
    print(f"Average waiting time: {total_waiting_time / num_steps:.2f}")
    print(f"Completion rate: {(total_arrived / total_vehicles * 100):.2f}%")
    print(f"Final action distribution: {action_counts}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_simulation.py <episode_number> [--no-gui]")
        sys.exit(1)
    
    episode = int(sys.argv[1])
    gui = "--no-gui" not in sys.argv
    
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    run_simulation(model_dir, episode, num_steps=500, gui=gui) 
