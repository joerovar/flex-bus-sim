import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from params import N_VEHICLES, TRIP_WEIGHT, STEPS_PER_EPISODE
from rl_env import FlexSimEnv  # Assuming FlexSimEnv is defined in rl_env.py
from itertools import product

# Define the experience tuple structure - include vehicle index for monitoring
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'veh_idx'])

class SharedReplayBuffer:
    """Replay buffer that pools experiences from multiple vehicles."""
    def __init__(self, buffer_size=10000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done, veh_idx):
        """Add an experience to the buffer with vehicle index."""
        experience = Experience(state, action, reward, next_state, done, veh_idx)
        self.buffer.append(experience)
    
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        
        # Convert observation dictionaries to arrays first
        states_array = np.array([self._dict_to_array(e.state) for e in experiences])
        next_states_array = np.array([self._dict_to_array(e.next_state) for e in experiences])
        
        # Convert to PyTorch tensors efficiently (from numpy arrays, not lists)
        states = torch.FloatTensor(states_array)
        next_states = torch.FloatTensor(next_states_array)
        actions = torch.LongTensor(np.array([[e.action] for e in experiences]))
        rewards = torch.FloatTensor(np.array([[e.reward] for e in experiences]))
        dones = torch.FloatTensor(np.array([[e.done] for e in experiences]))
        veh_indices = torch.LongTensor(np.array([[e.veh_idx] for e in experiences]))
        
        return (states, actions, rewards, next_states, dones, veh_indices)
    
    def _dict_to_array(self, obs_dict):
        """Convert observation dictionary to flat array for network input."""
        return np.concatenate([
            obs_dict["control_stop_idx"],
            obs_dict["n_requests"], 
            obs_dict["headway"],
            obs_dict["schedule_deviation"]
        ])
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network model - shared for all vehicles."""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SharedDQNAgent:
    """DQN Agent with a single policy shared across all vehicles."""
    def __init__(self, state_size=4, action_size=2, hidden_size=64, 
                 learning_rate=5e-3, gamma=0.99, buffer_size=1000,
                 batch_size=64, update_every=12, n_steps=1_000, 
                 training_portion=0.7): # TODO: make sure that you are calling it what it should be
        
        # Initialize a single shared Q-Network for all vehicles
        self.qnetwork = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.qnetwork.state_dict())
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        
        # Shared experience replay buffer
        self.memory = SharedReplayBuffer(buffer_size, batch_size)
        
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma

        epsilon_start = 1.0
        epsilon_min = 0.05
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min

        # linear decay
        n_exploration_steps = int(n_steps*training_portion)
        epsilon_decay_step = (self.epsilon-epsilon_min)/n_exploration_steps
        self.epsilon_decay = epsilon_decay_step

        self.update_every = update_every
        self.t_step = 0
        self.total_steps = 0
        
    def step(self, state, action, reward, next_state, done, veh_idx):
        """Process a step from the environment."""
        # Add experience to shared memory
        self.memory.add(state, action, reward, next_state, done, veh_idx)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        self.total_steps += 1
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
    
    def act(self, state, veh_idx=None, eval_mode=False):
        """
        Return action for given state as per current policy.
        veh_idx is accepted but ignored for action selection since policy is shared.
        It's kept for consistency with the environment interface.
        """
        # Check if n_requests is 0, in which case we always take action 0
        if state["n_requests"][0] == 0:
            return 0
        
        state_array = np.concatenate([
            state["control_stop_idx"],
            state["n_requests"], 
            state["headway"],
            state["schedule_deviation"]
        ])
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        
        # Use epsilon-greedy policy
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        # Get action from the shared network
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state_tensor)
        self.qnetwork.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self):
        """Update value parameters using batch of experience tuples."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from the shared buffer
        states, actions, rewards, next_states, dones, _ = self.memory.sample()
        
        # Get max predicted Q values for next states from target model
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.qnetwork(states).gather(1, actions)
        
        # Compute loss
        loss = nn.MSELoss()(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork, self.target_network)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
    
    def _soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update target network parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MultiVehicleTrainer:
    """Trainer that handles multiple vehicles in the environment with a shared policy."""
    def __init__(self, env, agent, num_episodes=200, n_steps=10_000, verbose=False):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_n_steps = n_steps
        self.scores = []
        self.verbose = verbose
        
    def train(self):
        """Run the training loop."""
        for episode in range(self.num_episodes):
            if self.agent.total_steps > self.max_n_steps:
                return self.scores
            episode_rewards = {i: 0 for i in range(N_VEHICLES)}
            episode_counts = {i: 0 for i in range(N_VEHICLES)}
            
            # Initialize state tracking for each vehicle
            vehicle_observations = {}
            vehicle_actions = {}  # Track previous actions
            
            # Start the episode
            next_observation, info = self.env.reset()
            vehicle_idx = info['veh_idx']

            # update observation
            observation = next_observation
            vehicle_observations[vehicle_idx] = observation
            
            # select action
            action = self.agent.act(observation, vehicle_idx)
            vehicle_actions[vehicle_idx] = action

            # take action in environment
            next_observation, reward, done, terminated, info = self.env.step(action)

            while not terminated:
                # Get current vehicle index
                vehicle_idx = info['veh_idx']
                
                # Check if this is a new vehicle
                is_first_appearance = vehicle_idx not in vehicle_observations

                if not is_first_appearance:
                    # add to the training buffer
                    observation = vehicle_observations[vehicle_idx]
                    if observation['control_stop_idx'] != 2:
                        action = vehicle_actions[vehicle_idx]
                        self.agent.step(observation, action, reward, next_observation, done, vehicle_idx)
                        episode_rewards[vehicle_idx] += reward
                        episode_counts[vehicle_idx] += 1
                
                # update observation
                observation = next_observation
                vehicle_observations[vehicle_idx] = observation

                # Select action using shared policy
                if not done:
                    action = self.agent.act(observation, vehicle_idx)
                    vehicle_actions[vehicle_idx] = action
                else:
                    action = None
                
                # Take action in environment
                next_observation, reward, done, terminated, info = self.env.step(action)
            
            # Track episode score (average across active vehicles)
            episode_score = sum(episode_rewards.values()) / sum(episode_counts.values())
            self.scores.append(episode_score)
            
            # Print progress
            if episode % 40 == 0 and self.verbose:
                print("------")
                print(f"Episode {episode}/{self.num_episodes}, Avg Score: {np.mean(self.scores[-40:]):.2f}, Epsilon: {self.agent.epsilon:.4f}, Number of steps: {self.agent.total_steps}")
                print("------")
        return self.scores

def evaluate_agent(env, agent, num_episodes=10, output_history=False,
                   scenario_name=None):
    """Evaluate the agent's performance in the environment."""
    rewards_per_episode = []
    results_per_episode = {
        'deviation_opportunities': [],
        'deviations': [],
        'avg_picked_requests': [],
        'early_trips': [],
        'late_trips': []
    }
    history = {'pax': [], 'vehicles': [],  'idle': []}

    for episode in range(num_episodes):
        # Start the episode
        next_observation, info = env.reset()

        # update observation
        observation = next_observation
        
        # select action
        action = agent.act(observation, eval_mode=True)

        # take action in environment
        next_observation, reward, done, terminated, info = env.step(action)
        rewards_per_episode.append(reward)

        while not terminated:
            if not done:
                action = agent.act(observation, eval_mode=True)
            else:
                action = None
            observation, reward, done, terminated, info = env.step(action=action)
            rewards_per_episode.append(reward)
            # # update observation
            # observation = next_observation

            # # Select action using shared policy
            
            # # Take action in environment
            # next_observation, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
        
        # recordings
        if output_history:
            episode_history = env.env.get_history()
            for key in episode_history:
                episode_history[key]['episode'] = episode
                if scenario_name:
                    episode_history[key]['scenario'] = scenario_name
                history[key].append(episode_history[key])

        # update results
        deviation_opps, deviations, avg_picked_requests, early_trips, late_trips = env.env.get_tracker_info()
        results_per_episode['deviation_opportunities'].append(deviation_opps)
        results_per_episode['deviations'].append(deviations)
        results_per_episode['avg_picked_requests'].append(avg_picked_requests)
        results_per_episode['early_trips'].append(early_trips)
        results_per_episode['late_trips'].append(late_trips)

    summary_results = {
        'mean_reward': round(np.nanmean(rewards_per_episode),3),
        'std_reward': round(np.nanstd(rewards_per_episode),3),
        'deviation_opportunities': round(np.mean(results_per_episode['deviation_opportunities']),1),
        'deviations': round(np.mean(results_per_episode['deviations']),1),
        'avg_picked_requests': round(np.mean(results_per_episode['avg_picked_requests']),1),
        'early_trips': round(np.mean(results_per_episode['early_trips']),1),
        'late_trips': round(np.mean(results_per_episode['late_trips']),1)
    }
    if output_history:
        for df_key in history:
            history[df_key] = pd.concat(history[df_key])
        return history, summary_results
    else:
        return summary_results

def train_vehicles(reward_weight=TRIP_WEIGHT,
                   learning_rate=5e-4, 
                   gamma=0.99, buffer_size=1000,
                   batch_size=64, update_every=12, 
                   n_steps=1_000, save_path=None):
    env = FlexSimEnv(reward_weight=reward_weight)
    state_size = 4  # control_stop_idx, n_requests, headway, schedule_deviation
    action_size = 2  # binary action
    num_episodes = int(n_steps/STEPS_PER_EPISODE) + 5 # extra episodes as buffer
    
    agent = SharedDQNAgent(state_size=state_size, action_size=action_size,
                           learning_rate=learning_rate, 
                           gamma=gamma, buffer_size=buffer_size,
                           batch_size=batch_size, 
                           update_every=update_every, n_steps=n_steps)
    trainer = MultiVehicleTrainer(env, agent, num_episodes=num_episodes)
    
    print("Training with shared policy...")
    scores = trainer.train()
    
    print(f"\nTraining complete! Final average reward: {np.mean(scores[-40:]):.2f}\n")

    # save
    if save_path:
        torch.save(agent.qnetwork.state_dict(), save_path)
        print(f"Model saved as {save_path}")
    return agent, scores

def load_agent(model_path):
    # Initialize a new agent with the same architecture
    state_size = 4
    action_size = 2
    agent = SharedDQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the saved state dictionary with weights_only=True to address the warning
    agent.qnetwork.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Set to evaluation mode (disables dropout, etc.)
    agent.qnetwork.eval()
    return agent

def grid_search_dqn(
    lr_values=[5e-3, 6e-3], # learning rate
    n_steps_values=[12_000, 24_000], # timesteps
    gamma_values=[0.98, 0.99], # discount factor
    reward_weights=[TRIP_WEIGHT],
    verbose=False
):  
    # Dictionary to store results
    grid_search_results = {
        'weight': [],
        'lr': [],
        'n_steps': [],
        'gamma': [],
        'mean_reward': [],
        'std_reward': [],
        'deviation_opportunities': [],
        'deviations': [],
        'avg_picked_requests': [],
        'early_trips': [],
        'late_trips': []
    }
    
    # Grid search
    for lr, n_steps, gamma, weight in product(lr_values, n_steps_values, gamma_values, reward_weights):
        # Train model with current parameters
        np.random.seed(0)

        agent, scores = train_vehicles(learning_rate=lr, n_steps=n_steps, gamma=gamma, reward_weight=weight)
        
        # Evaluate model
        env = FlexSimEnv(reward_weight=weight)
        summary_results = evaluate_agent(env, agent, num_episodes=30)
        
        ## Store in results
        grid_search_results['weight'].append(weight)
        grid_search_results['lr'].append(lr)
        grid_search_results['n_steps'].append(n_steps)       
        grid_search_results['gamma'].append(gamma)

        for key in summary_results:
            grid_search_results[key].append(summary_results[key])

        if verbose:
            print(f"Evaluation summary:")
            print(f"Params: lr={lr}, n_steps={n_steps}, gamma={gamma}, reward_weight={weight}")
            print(f"Reward: {round(summary_results['mean_reward'], 3)} +/- {round(summary_results['std_reward'], 3)}")
            print("------------------------")    
    return grid_search_results

