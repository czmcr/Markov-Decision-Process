import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Ensure TensorFlow uses appropriate memory management
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Environment setup
GRID_SIZE = 5
NUM_AGENTS = 4
A_LOCATION = (1, 1)
B_LOCATION = (4, 4)

# Actions: 0=North, 1=South, 2=West, 3=East
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores transitions (state, action, reward, next_state, done) 
    for experience replay during training.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for grid world navigation.
    Uses a neural network to approximate the Q-function.
    """
    def __init__(self, idx, state_size, action_size=4):
        self.idx = idx
        self.reset()
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate
        
        # Neural networks (main and target)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target network
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=5000)
        self.batch_size = 32
        
        # Performance tracking
        self.rewards_log = []
        self.deliveries = 0
        self.step_count = 0
        self.update_counter = 0
        self.target_update_freq = 100  # Update target network every 100 steps
    
    def _build_model(self):
        """Build a neural network for Q-function approximation"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target network with weights from main network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def reset(self):
        """Sets initial position and carrying status for off-the-job training."""
        self.pos = A_LOCATION if self.idx % 2 == 0 else B_LOCATION
        self.carrying = self.pos == A_LOCATION
    
    def get_state_vector(self, grid, agents):
        """
        Convert the agent's state into a vector representation for the neural network.
        """
        # Agent's position (normalized)
        pos_x = self.pos[0] / (GRID_SIZE - 1)
        pos_y = self.pos[1] / (GRID_SIZE - 1)
        
        # Carrying state (0 or 1)
        carrying = 1.0 if self.carrying else 0.0
        
        # Target location (A or B depending on carrying state)
        target = B_LOCATION if self.carrying else A_LOCATION
        target_x = target[0] / (GRID_SIZE - 1)
        target_y = target[1] / (GRID_SIZE - 1)
        
        # Direction to target (normalized)
        dir_x = (target[0] - self.pos[0]) / GRID_SIZE
        dir_y = (target[1] - self.pos[1]) / GRID_SIZE
        
        # Distance to target (normalized)
        dist_x = abs(target[0] - self.pos[0]) / GRID_SIZE
        dist_y = abs(target[1] - self.pos[1]) / GRID_SIZE
        
        # Neighboring agents' info (check all 8 directions)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        neighbor_features = []
        for dx, dy in directions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                agent_id = grid[nx][ny]
                if agent_id != -1:
                    other_agent = agents[agent_id]
                    # 1 if opposite carrying state, 0 otherwise
                    opposite_type = 1.0 if other_agent.carrying != self.carrying else 0.0
                    neighbor_features.append(opposite_type)
                else:
                    neighbor_features.append(0.0)  # No agent
            else:
                neighbor_features.append(-1.0)  # Wall
                
        # Location flags (special positions)
        at_A = 1.0 if self.pos == A_LOCATION else 0.0
        at_B = 1.0 if self.pos == B_LOCATION else 0.0
        
        # Combine all features
        state = [pos_x, pos_y, carrying, 
                 target_x, target_y, dir_x, dir_y, 
                 dist_x, dist_y, at_A, at_B] + neighbor_features
                 
        return np.array(state)
    
    def choose_action(self, grid, agents, training=True):
        state = self.get_state_vector(grid, agents)
        
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Use model for prediction
        state_tensor = np.reshape(state, (1, self.state_size))
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """Train model using experience replay"""
        if self.memory.size() < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Get target Q values from target model
        target_q = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.max(target_q, axis=1) * (1 - dones)
        
        # Get current Q values and update targets for actions taken
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Update target network if needed
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def move(self, action, grid):
        """Move agent and handle collision detection."""
        dx, dy = actions[action]
        new_x = min(max(self.pos[0] + dx, 0), GRID_SIZE - 1)
        new_y = min(max(self.pos[1] + dy, 0), GRID_SIZE - 1)
        
        # Record previous state for reward calculation
        old_pos = self.pos
        old_carrying = self.carrying
        
        # Update position
        self.pos = (new_x, new_y)
        
        # Handle pickup/dropoff
        if self.pos == A_LOCATION:
            if not self.carrying:
                self.deliveries += 1  # Count completed cycle
            self.carrying = True
        elif self.pos == B_LOCATION:
            self.carrying = False
            
        self.step_count += 1
        return old_pos, old_carrying
    
    def save(self, filepath):
        """Save the trained model"""
        save_model(self.model, f"{filepath}_agent{self.idx}.h5")
    
    def load(self, filepath):
        """Load a trained model"""
        self.model = load_model(f"{filepath}_agent{self.idx}.h5")
        self.update_target_model()

def head_on_collision(agent, other):
    """Check if two agents have a head-on collision."""
    return agent.pos == other.pos and agent.carrying != other.carrying

def calculate_state_size():
    """Calculate the dimension of the state vector"""
    # Position (2) + carrying (1) + target location (2) + direction to target (2) + 
    # distance to target (2) + at A/B flags (2) + 8 neighboring cells
    return 2 + 1 + 2 + 2 + 2 + 2 + 8

# Initialize agents
state_size = calculate_state_size()
agents = [DQNAgent(i, state_size) for i in range(NUM_AGENTS)]

# Training settings
max_episodes = 10000
max_steps_per_episode = 25
step_budget = 1_500_000
collision_budget = 4000
walltime_budget = 600  # 10 minutes in seconds

def train_agents():
    """Train the agents using DQN with experience replay"""
    start_time = time.time()
    total_steps = 0
    collision_count = 0
    episode = 0
    
    # For monitoring progress
    eval_interval = 100
    rewards_history = []
    collision_history = []
    success_rate_history = []
    
    print("Training agents with DQN and step/collision/time budget...")
    
    # Create a schedule for off-the-job training
    training_schedules = [
        # Schedule 1: All agents start at their default positions
        lambda: [agent.reset() for agent in agents],
        
        # Schedule 2: Agents start in corners
        lambda: [setattr(agents[i], 'pos', pos) for i, pos in 
                 enumerate([(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE-1)])],
                 
        # Schedule 3: Agents start at random positions
        lambda: [setattr(agents[i], 'pos', (random.randint(0, GRID_SIZE-1), 
                                          random.randint(0, GRID_SIZE-1))) for i in range(NUM_AGENTS)]
    ]
    
    while (total_steps < step_budget and 
           collision_count < collision_budget and 
           (time.time() - start_time) < walltime_budget and
           episode < max_episodes):
        
        # Select a training schedule for this episode
        schedule_idx = episode % len(training_schedules)
        training_schedules[schedule_idx]()
        
        # Set carrying status based on position
        for agent in agents:
            agent.carrying = agent.pos == A_LOCATION
        
        # Initialize grid
        grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for agent in agents:
            grid[agent.pos[0]][agent.pos[1]] = agent.idx
            
        episode_reward = 0
        episode_collisions = 0
        
        for step in range(max_steps_per_episode):
            for agent_idx, agent in enumerate(agents):  # Round-robin update
                if total_steps >= step_budget:
                    break
                    
                total_steps += 1
                
                # Get current state vector
                state = agent.get_state_vector(grid, agents)
                
                # Choose and perform action
                action = agent.choose_action(grid, agents, training=True)
                
                # Clear agent's old position in grid
                grid[agent.pos[0]][agent.pos[1]] = -1
                
                # Move agent and get old state for reward calculation
                old_pos, old_carrying = agent.move(action, grid)
                
                # Update grid with new position
                grid[agent.pos[0]][agent.pos[1]] = agent.idx
                
                # Calculate reward
                reward = 0
                done = False
                
                # Reward for successful delivery
                if old_pos == A_LOCATION and agent.pos != A_LOCATION and old_carrying:
                    reward += 0.1  # Small reward for leaving A with item
                
                if old_pos == B_LOCATION and agent.pos != B_LOCATION and not old_carrying:
                    reward += 0.1  # Small reward for leaving B after dropoff
                
                # Large reward for successful delivery
                if old_carrying and agent.pos == B_LOCATION:
                    reward += 1.0
                
                # Even larger reward for completing a cycle (B→A→B)
                if not old_carrying and agent.pos == A_LOCATION:
                    reward += 1.5
                    done = True  # Consider a completed cycle as a terminal state
                
                # Check for collisions
                for other in agents:
                    if other.idx != agent.idx and head_on_collision(agent, other):
                        reward -= 10.0  # Strong penalty for collision
                        collision_count += 1
                        episode_collisions += 1
                        done = True  # Consider a collision as a terminal state
                
                # Small penalty for each step to encourage efficiency
                reward -= 0.01
                
                # Get new state
                next_state = agent.get_state_vector(grid, agents)
                
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from past experiences
                agent.replay()
                
                # Track rewards
                agent.rewards_log.append(reward)
                episode_reward += reward
        
        # End of episode
        episode += 1
        rewards_history.append(episode_reward / NUM_AGENTS)
        collision_history.append(episode_collisions)
        
        # Periodically evaluate and print progress
        if episode % eval_interval == 0:
            success_rate = evaluate_agents(silent=True)
            success_rate_history.append(success_rate)
            
            elapsed_time = time.time() - start_time
            avg_epsilon = sum(agent.epsilon for agent in agents) / NUM_AGENTS
            print(f"Episode {episode}: Steps={total_steps}, Collisions={collision_count}, "
                  f"Time={elapsed_time:.1f}s, Success Rate={success_rate:.1f}%, "
                  f"Epsilon={avg_epsilon:.3f}, Avg Reward={rewards_history[-1]:.2f}")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed:")
    print(f"Episodes={episode}, Steps={total_steps}, Collisions={collision_count}")
    print(f"Time Elapsed={elapsed_time:.2f}s")
    
    # Plot learning curves
    plot_learning_curves(rewards_history, collision_history, success_rate_history, eval_interval)
    
    return total_steps, collision_count, elapsed_time

def plot_learning_curves(rewards, collisions, success_rates, eval_interval):
    """Plot the learning progress of the agents."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot average rewards
    episodes = range(len(rewards))
    ax1.plot(episodes, rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward per Episode')
    ax1.grid(True)
    
    # Plot collision count
    ax2.plot(episodes, collisions)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Collisions')
    ax2.set_title('Collisions per Episode')
    ax2.grid(True)
    
    # Plot success rate (evaluated every eval_interval episodes)
    eval_episodes = range(0, len(rewards), eval_interval)
    eval_episodes = eval_episodes[:len(success_rates)]  # Ensure lengths match
    ax3.plot(eval_episodes, success_rates)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate During Training')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_learning_curves.png')
    plt.show()

def evaluate_agents(num_runs=100, silent=False):
    """Evaluate the agents' performance."""
    successful_runs = 0
    collision_free_runs = 0
    
    for _ in range(num_runs):
        # Reset agents
        for agent in agents:
            agent.reset()
        
        # Initialize grid
        grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for agent in agents:
            grid[agent.pos[0]][agent.pos[1]] = agent.idx
        
        steps_taken = 0
        local_collisions = 0
        delivery_success = [False] * NUM_AGENTS
        
        for step in range(25):  # Max steps for evaluation
            steps_taken += 1
            
            for agent in agents:
                # Get current state
                state = agent.get_state_vector(grid, agents)
                
                # Choose action (no exploration during evaluation)
                state_tensor = np.reshape(state, (1, state_size))
                q_values = agent.model.predict(state_tensor, verbose=0)[0]
                action = np.argmax(q_values)
                
                # Clear old position
                grid[agent.pos[0]][agent.pos[1]] = -1
                
                # Move agent
                old_pos = agent.pos
                agent.move(action, grid)
                
                # Update grid
                grid[agent.pos[0]][agent.pos[1]] = agent.idx
                
                # Check for delivery success
                if old_pos == A_LOCATION and agent.pos == B_LOCATION:
                    delivery_success[agent.idx] = True
                
                # Check for collisions
                for other in agents:
                    if other.idx != agent.idx and head_on_collision(agent, other):
                        local_collisions += 1
        
        if all(delivery_success) and steps_taken <= 25:
            successful_runs += 1
        if local_collisions == 0:
            collision_free_runs += 1
    
    success_rate = (successful_runs / num_runs) * 100
    collision_free_rate = (collision_free_runs / num_runs) * 100
    
    if not silent:
        print(f"--- Evaluation Results ---")
        print(f"Success Rate (all deliveries in ≤25 steps): {success_rate:.2f}%")
        print(f"Collision-Free Runs: {collision_free_rate:.2f}%")
    
    return success_rate

# Train the agents
total_steps, collision_count, elapsed_time = train_agents()

# Final evaluation
success_rate = evaluate_agents(num_runs=100)

# Save trained models
for agent in agents:
    agent.save("dqn_model")

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))

def draw_grid():
    """Draws the static grid with A and B labeled."""
    ax.clear()
    ax.set_xticks(np.arange(0.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(0.5, GRID_SIZE, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.invert_yaxis()
    ax.text(A_LOCATION[1], A_LOCATION[0], 'A', ha='center', va='center', fontsize=12, color='green')
    ax.text(B_LOCATION[1], B_LOCATION[0], 'B', ha='center', va='center', fontsize=12, color='blue')
    ax.set_title("Multi-Agent Delivery: A ↔ B", fontsize=14)

def update(frame):
    """
    Updates the grid visualization and performs one step for each agent.
    Uses trained DQN models for decision making.
    """
    draw_grid()
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for agent in agents:
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    for agent in agents:  # Round-robin scheduling
        # Get current state
        state = agent.get_state_vector(grid, agents)
        
        # Choose action using trained model
        state_tensor = np.reshape(state, (1, state_size))
        q_values = agent.model.predict(state_tensor, verbose=0)[0]
        action = np.argmax(q_values)
        
        # Clear old position
        grid[agent.pos[0]][agent.pos[1]] = -1
        
        # Move agent
        agent.move(action, grid)
        
        # Update grid
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    # Plot agent positions
    for agent in agents:
        color = 'green' if agent.carrying else 'blue'
        label = f"A{agent.idx} {'[Carrying]' if agent.carrying else '[Empty]'}"
        ax.plot(agent.pos[1], agent.pos[0], 'o', markersize=12, color=color)
        ax.text(agent.pos[1], agent.pos[0], f"{agent.idx}", ha='center', va='center', color='white', fontsize=8)
        ax.text(agent.pos[1], agent.pos[0] + 0.3, label, ha='center', va='bottom', fontsize=6)

    fig.canvas.draw()

# For demonstration, show the agents in action
for agent in agents:
    agent.reset()  # Reset agents for visualization

ani = FuncAnimation(fig, update, frames=200, interval=300, repeat=False)
plt.show()