# Multi-Agent Q-Learning Environment for FIT5226 Project
# With Enhanced Metrics Visualization, Robustness Testing, and Collision Analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pickle
import time
from collections import defaultdict, deque
import seaborn as sns

# Environment setup
GRID_SIZE = 5
NUM_AGENTS = 4
DEFAULT_A_LOCATION = (1, 1)
DEFAULT_B_LOCATION = (4, 4)

# Actions: 0=North, 1=South, 2=West, 3=East
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_names = ["North", "South", "West", "East"]

class Agent:
    """
    Represents an agent in the grid world with a simple Q-table.
    Each agent learns to move from A to B and back while avoiding head-on collisions.
    Includes use of:
      - State of neighboring cells (only considers agents of opposite type)
      - Central clock (update schedule is round-robin)
      - Off-the-job training (start configs defined below)
    """
    def __init__(self, idx, shared_q=None):
        self.idx = idx
        self.reset()
        # Use shared Q-table if provided, otherwise create individual Q-table
        self.q_table = shared_q if shared_q is not None else {}
        self.epsilon = 0.2
        self.alpha = 1e-3
        self.gamma = 0.99
        self.rewards_log = []
        self.last_visited = []  # Track recently visited positions
        self.max_visit_history = 5

    def reset(self, a_location=DEFAULT_A_LOCATION, b_location=DEFAULT_B_LOCATION):
        """Sets initial position and carrying status for off-the-job training."""
        self.pos = a_location if self.idx % 2 == 0 else b_location
        self.carrying = self.pos == a_location
        self.last_visited = []
        self.a_loc = a_location
        self.b_loc = b_location

    def reset_at_location(self, location, carrying=False):
        """Reset agent to a specific location with specified carrying status."""
        self.pos = location
        self.carrying = carrying
        self.last_visited = []

    def get_neighbors_state(self, grid, agents):
        """
        Returns a binary string where 1 means an adjacent cell has an agent
        of opposite carrying state (e.g., one is carrying and the other is not).
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E only
        bits = ''
        for dx, dy in directions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                agent_id = grid[nx][ny]
                if agent_id != -1:
                    other_agent = agents[agent_id]
                    if other_agent.carrying != self.carrying:
                        bits += '1'
                        continue
            bits += '0'
        return bits

    def get_state(self, grid, agents):
        neighbors = self.get_neighbors_state(grid, agents)
        return (self.pos[0], self.pos[1], int(self.carrying), neighbors)

    def choose_action(self, grid, agents, eval_mode=False):
        state = self.get_state(grid, agents)
        
        # Initialize Q-values if state is new
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Add slight preference for unexplored paths during evaluation
        if eval_mode and len(self.last_visited) > 0:
            q_values = np.array(self.q_table[state])
            # Penalize recently visited positions
            for action in range(4):
                dx, dy = actions[action]
                new_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if new_pos in self.last_visited:
                    q_values[action] -= 0.1  # Small penalty
            return int(np.argmax(q_values))
        
        return int(np.argmax(self.q_table[state]))

    def update_q(self, prev_state, action, reward, next_state):
        if prev_state not in self.q_table:
            self.q_table[prev_state] = [0.0] * 4
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 4
        max_next = max(self.q_table[next_state])
        self.q_table[prev_state][action] += self.alpha * (reward + self.gamma * max_next - self.q_table[prev_state][action])

    def move(self, action):
        dx, dy = actions[action]
        new_x = min(max(self.pos[0] + dx, 0), GRID_SIZE - 1)
        new_y = min(max(self.pos[1] + dy, 0), GRID_SIZE - 1)
        
        # Update visit history (for evaluation mode path diversity)
        if len(self.last_visited) >= self.max_visit_history:
            self.last_visited.pop(0)  # Remove oldest
        self.last_visited.append(self.pos)
        
        self.pos = (new_x, new_y)

    def update_carrying_status(self, a_location=DEFAULT_A_LOCATION, b_location=DEFAULT_B_LOCATION):
        """Updates carrying status based on current position"""
        # When at A, pickup supply (become carrying)
        if self.pos == a_location and not self.carrying:
            self.carrying = True
            return 1  # Reward for pickup
        # When at B, deliver supply (become not carrying)
        elif self.pos == b_location and self.carrying:
            self.carrying = False
            return 1  # Reward for delivery
        return 0  # No reward if no pickup/delivery happened

class Environment:
    """Manages the grid world environment, agents, and metrics collection."""
    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS, a_loc=DEFAULT_A_LOCATION, b_loc=DEFAULT_B_LOCATION):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.a_location = a_loc
        self.b_location = b_loc
        self.grid = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Metrics storage
        self.episode_rewards = []
        self.collision_locations = defaultdict(int)
        self.step_counts = []
        self.delivery_counts = []
        self.collision_counts = []
        self.avg_rewards = []
        self.success_rates = deque(maxlen=100)  # Recent success rate
        
        # Shared Q-table for all agents
        self.shared_q_table = {}
        self.agents = [Agent(i, shared_q=self.shared_q_table) for i in range(num_agents)]
        
        # Training budgets
        self.step_budget = 1_500_000
        self.collision_budget = 4000
        self.walltime_budget = 600  # 10 minutes
        
        # Initialize counters
        self.total_steps = 0
        self.collision_count = 0
        self.deliveries_completed = 0
        self.start_time = None

    def reset(self, a_loc=None, b_loc=None, testing=False):
        """Reset the environment with optional new A and B locations."""
        if a_loc is not None:
            self.a_location = a_loc
        if b_loc is not None:
            self.b_location = b_loc
            
        self.grid = [[-1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        if testing:
            # For testing, initialize all agents at B
            for agent in self.agents:
                agent.reset_at_location(self.b_location, carrying=False)
        else:
            # For training, use the regular initialization
            for agent in self.agents:
                agent.reset(self.a_location, self.b_location)
        
        # Update grid with agent positions
        for agent in self.agents:
            self.grid[agent.pos[0]][agent.pos[1]] = agent.idx
            
        # Reset episode metrics
        self.episode_step_count = 0
        self.episode_collision_count = 0
        self.episode_delivery_count = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]

    def head_on_collision(self, agent, other):
        """Check if two agents have a head-on collision."""
        return agent.pos == other.pos and agent.carrying != other.carrying

    def run_training_episode(self, max_steps=20):
        """Run a single training episode."""
        for step in range(max_steps):
            self.episode_step_count += 1
            
            # Process agents in round-robin order (central clock)
            for agent in self.agents:
                self.total_steps += 1
                
                # Get current state
                state = agent.get_state(self.grid, self.agents)
                
                # Choose action
                action = agent.choose_action(self.grid, self.agents)
                
                # Proposed move coordinates
                dx, dy = actions[action]
                proposed_x = min(max(agent.pos[0] + dx, 0), self.grid_size - 1)
                proposed_y = min(max(agent.pos[1] + dy, 0), self.grid_size - 1)
                
                # Check if move is allowed
                occupying_agent_id = self.grid[proposed_x][proposed_y]
                move_allowed = True
                
                if occupying_agent_id != -1:
                    occupying_agent = self.agents[occupying_agent_id]
                    if occupying_agent.carrying != agent.carrying:
                        # Discourage illegal move into conflicting traffic
                        reward = -1
                        agent.rewards_log.append(reward)
                        self.episode_rewards[agent.idx].append(reward)
                        move_allowed = False
                
                if move_allowed:
                    # Clear the agent's position in the grid
                    self.grid[agent.pos[0]][agent.pos[1]] = -1
                    old_pos = agent.pos
                    
                    # Execute move
                    agent.move(action)
                    
                    # Initialize reward
                    reward = 0
                    
                    # Add reward for pickup or delivery
                    delivery_reward = agent.update_carrying_status(self.a_location, self.b_location)
                    reward += delivery_reward
                    if delivery_reward > 0 and agent.pos == self.b_location:
                        self.deliveries_completed += 1
                        self.episode_delivery_count += 1
                    
                    # Check for collisions
                    collision_detected = False
                    for other in self.agents:
                        if other is not agent and self.head_on_collision(agent, other):
                            reward = -10  # Penalize head-on collision
                            self.collision_count += 1
                            self.episode_collision_count += 1
                            collision_detected = True
                            # Record collision location for analysis
                            self.collision_locations[agent.pos] += 1
                    
                    # Update reward logs
                    agent.rewards_log.append(reward)
                    self.episode_rewards[agent.idx].append(reward)
                    
                    # Update Q-table
                    next_state = agent.get_state(self.grid, self.agents)
                    agent.update_q(state, action, reward, next_state)
                    
                    # Update agent position in grid
                    self.grid[agent.pos[0]][agent.pos[1]] = agent.idx

        # Calculate episode metrics
        avg_reward = np.mean([np.sum(rewards) for rewards in self.episode_rewards])
        self.avg_rewards.append(avg_reward)
        self.step_counts.append(self.episode_step_count)
        self.collision_counts.append(self.episode_collision_count)
        self.delivery_counts.append(self.episode_delivery_count)
        
        # Check if episode was successful (at least one delivery with no collisions)
        success = self.episode_delivery_count > 0 and self.episode_collision_count == 0
        self.success_rates.append(1 if success else 0)
        
        return avg_reward, self.episode_delivery_count, self.episode_collision_count

    def train(self, num_episodes=None, progress_interval=100):
        """Train agents until budget constraints are reached."""
        print("Starting training with budget constraints:")
        print(f"  Step budget: {self.step_budget}")
        print(f"  Collision budget: {self.collision_budget}")
        print(f"  Walltime budget: {self.walltime_budget} seconds")
        
        self.start_time = time.time()
        episode = 0
        
        while True:
            # Check if we've hit any budget limits
            elapsed_time = time.time() - self.start_time
            if self.total_steps >= self.step_budget:
                print("\nTraining stopped: Step budget exceeded")
                break
            if self.collision_count >= self.collision_budget:
                print("\nTraining stopped: Collision budget exceeded")
                break
            if elapsed_time >= self.walltime_budget:
                print("\nTraining stopped: Time budget exceeded")
                break
            if num_episodes is not None and episode >= num_episodes:
                print("\nTraining stopped: Episode limit reached")
                break
                
            # Run one training episode
            self.reset()
            avg_reward, deliveries, collisions = self.run_training_episode()
            
            # Print progress periodically
            if episode % progress_interval == 0:
                success_rate = np.mean(self.success_rates) * 100 if self.success_rates else 0
                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, " 
                      f"Deliveries={deliveries}, Collisions={collisions}, "
                      f"Steps={self.total_steps}, Recent Success={success_rate:.1f}%")
            
            episode += 1
            
            # Adjust exploration rate over time (optional annealing)
            if episode % 1000 == 0:
                for agent in self.agents:
                    agent.epsilon = max(0.05, agent.epsilon * 0.95)  # Gradually reduce exploration
        
        elapsed_time = time.time() - self.start_time
        print(f"\nTraining completed: Episodes={episode}, Steps={self.total_steps}, "
              f"Collisions={self.collision_count}, Deliveries={self.deliveries_completed}, "
              f"Time={elapsed_time:.2f}s")
        
        # Save final metrics
        self.training_episodes = episode
        self.training_time = elapsed_time
        
        return {
            'episodes': episode,
            'steps': self.total_steps,
            'collisions': self.collision_count,
            'deliveries': self.deliveries_completed,
            'time': elapsed_time
        }

    def evaluate_performance(self, num_trials=100, max_steps=25, configs=None):
        """
        Evaluate agent performance with optional different configurations.
        
        Args:
            num_trials: Number of evaluation trials
            max_steps: Maximum steps per trial
            configs: List of (a_loc, b_loc) configurations to test, or None for default
        
        Returns:
            Dictionary of evaluation results
        """
        print(f"\nEvaluating performance over {num_trials} trials ({max_steps} steps max)...")
        
        if configs is None:
            # Use default configuration for all trials
            configs = [(self.a_location, self.b_location)] * num_trials
        
        successful_trials = 0
        delivery_steps = []
        collision_trials = 0
        
        # Detailed results for analysis
        results = {
            'success_by_config': defaultdict(list),
            'steps_by_config': defaultdict(list),
            'collisions_by_config': defaultdict(int)
        }
        
        for trial, (a_loc, b_loc) in enumerate(configs):
            # Reset environment with this configuration
            self.reset(a_loc, b_loc, testing=True)
            
            # Track if this trial had any collisions
            collision_occurred = False
            # Track if agents completed a delivery in this trial
            delivery_completed = [False] * self.num_agents
            
            # Record steps taken for first delivery
            first_delivery_step = None
            
            # Run the trial for max_steps steps
            for step in range(max_steps):
                for agent_idx, agent in enumerate(self.agents):
                    # Get state and choose action (no exploration during evaluation)
                    state = agent.get_state(self.grid, self.agents)
                    action = agent.choose_action(self.grid, self.agents, eval_mode=True)
                    
                    # Clear agent's current position in grid
                    self.grid[agent.pos[0]][agent.pos[1]] = -1
                    
                    # Execute move
                    agent.move(action)
                    
                    # Check for delivery completion: B→A→B cycle
                    if agent.pos == a_loc and not agent.carrying:
                        agent.carrying = True  # Pick up at A
                    elif agent.pos == b_loc and agent.carrying:
                        agent.carrying = False  # Deliver at B
                        delivery_completed[agent_idx] = True
                        if first_delivery_step is None:
                            first_delivery_step = step
                    
                    # Check for collisions
                    for other in self.agents:
                        if other is not agent and self.head_on_collision(agent, other):
                            collision_occurred = True
                    
                    # Update grid with new position
                    self.grid[agent.pos[0]][agent.pos[1]] = agent.idx
                
                # If all agents completed delivery or collision occurred, end trial
                if all(delivery_completed) or collision_occurred:
                    break
            
            # Record results
            config_key = f"A{a_loc}-B{b_loc}"
            success = any(delivery_completed) and not collision_occurred
            
            if success:
                successful_trials += 1
                if first_delivery_step is not None:
                    delivery_steps.append(first_delivery_step)
            
            if collision_occurred:
                collision_trials += 1
                results['collisions_by_config'][config_key] += 1
            
            results['success_by_config'][config_key].append(success)
            if first_delivery_step is not None:
                results['steps_by_config'][config_key].append(first_delivery_step)
                
            # Print progress for long evaluations
            if (trial + 1) % 20 == 0:
                print(f"  Processed {trial + 1}/{num_trials} trials...")
        
        # Calculate overall metrics
        success_rate = successful_trials / num_trials * 100
        avg_steps = np.mean(delivery_steps) if delivery_steps else float('inf')
        collision_rate = collision_trials / num_trials * 100
        
        print(f"Performance results: {successful_trials}/{num_trials} successful trials ({success_rate:.2f}%)")
        print(f"Average successful delivery steps: {avg_steps:.2f}")
        print(f"Collision rate: {collision_rate:.2f}%")
        
        # Check if performance meets requirement (75% success rate)
        if success_rate >= 75:
            print("Performance requirement met (≥75% success rate)")
        else:
            print("Performance requirement not met (<75% success rate)")
        
        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'collision_rate': collision_rate,
            'successful_trials': successful_trials,
            'collision_trials': collision_trials,
            'detailed': results
        }

    def save_model(self, filename="q_tables.pkl"):
        """Save the trained Q-table."""
        with open(filename, "wb") as f:
            pickle.dump([self.shared_q_table], f)
        print(f"Model saved to {filename}")

    def load_model(self, filename="q_tables.pkl"):
        """Load a previously trained Q-table."""
        try:
            with open(filename, "rb") as f:
                q_tables = pickle.load(f)
                self.shared_q_table = q_tables[0]
                
                # Update agent Q-tables
                for agent in self.agents:
                    agent.q_table = self.shared_q_table
                    
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found")
            return False

    def visualize_metrics(self):
        """Generate visualizations of training metrics."""
        if not self.avg_rewards:
            print("No training data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Learning curve (rewards)
        smoothing_window = min(50, len(self.avg_rewards))
        if smoothing_window > 0:
            smoothed_rewards = np.convolve(self.avg_rewards, 
                                          np.ones(smoothing_window)/smoothing_window, 
                                          mode='valid')
            episodes = range(smoothing_window-1, len(self.avg_rewards))
            axes[0, 0].plot(episodes, smoothed_rewards)
            axes[0, 0].set_title('Learning Curve (Smoothed Rewards)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True)
        
        # Plot 2: Collision rate over training
        if len(self.collision_counts) > 0:
            window_size = min(100, len(self.collision_counts))
            collision_rate = []
            for i in range(window_size, len(self.collision_counts)+1):
                window = self.collision_counts[i-window_size:i]
                collision_rate.append(sum(1 for x in window if x > 0) / window_size)
            
            axes[0, 1].plot(range(window_size, len(self.collision_counts)+1), collision_rate)
            axes[0, 1].set_title('Collision Rate (Moving Average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Collision Rate')
            axes[0, 1].grid(True)
        
        # Plot 3: Success rate over time
        if len(self.success_rates) > 0:
            window_size = min(100, len(self.success_rates))
            success_rate = []
            for i in range(window_size, len(self.success_rates)+1):
                success_rate.append(np.mean(list(self.success_rates)[i-window_size:i]) * 100)
            
            axes[1, 0].plot(range(window_size, len(self.success_rates)+1), success_rate)
            axes[1, 0].set_title('Success Rate (Moving Average)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].grid(True)
            axes[1, 0].set_ylim(0, 100)
        
        # Plot 4: Delivery steps histogram
        if self.delivery_counts:
            axes[1, 1].hist(self.delivery_counts, bins=max(10, min(50, len(self.delivery_counts)//100)))
            axes[1, 1].set_title('Deliveries per Episode')
            axes[1, 1].set_xlabel('Deliveries')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
        # Create a heatmap of collision locations
        if self.collision_locations:
            plt.figure(figsize=(8, 6))
            collision_grid = np.zeros((self.grid_size, self.grid_size))
            for (x, y), count in self.collision_locations.items():
                collision_grid[x][y] = count
            
            # Plot heatmap
            sns.heatmap(collision_grid, annot=True, fmt=".0f", cmap="YlOrRd")
            plt.title('Collision Heatmap')
            plt.xlabel('Y coordinate')
            plt.ylabel('X coordinate')
            plt.savefig('collision_heatmap.png')
            plt.show()

    def visualize_agent_paths(self, max_steps=25):
        """Visualize the learned paths of agents on the grid."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Reset for visualization
        self.reset(testing=True)
        
        # Track paths for each agent
        paths = [[] for _ in range(self.num_agents)]
        for agent in self.agents:
            paths[agent.idx].append(agent.pos)
        
        # Run simulation and collect path data
        for step in range(max_steps):
            for agent in self.agents:
                state = agent.get_state(self.grid, self.agents)
                action = agent.choose_action(self.grid, self.agents, eval_mode=True)
                
                # Record action for analysis
                dx, dy = actions[action]
                
                # Clear old position
                self.grid[agent.pos[0]][agent.pos[1]] = -1
                
                # Move agent
                agent.move(action)
                
                # Update carrying status
                agent.update_carrying_status(self.a_location, self.b_location)
                
                # Update grid
                self.grid[agent.pos[0]][agent.pos[1]] = agent.idx
                
                # Record new position
                paths[agent.idx].append(agent.pos)
        
        # Setup the grid visualization
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.invert_yaxis()  # Invert y-axis to match grid coordinates
        ax.grid(True)
        
        # Mark A and B locations
        ax.plot(self.a_location[1], self.a_location[0], 'gs', markersize=20, alpha=0.5, label='A')
        ax.plot(self.b_location[1], self.b_location[0], 'bs', markersize=20, alpha=0.5, label='B')
        
        # Plot paths
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        for i, path in enumerate(paths):
            if not path:
                continue
                
            # Extract x and y coordinates
            xs = [p[1] for p in path]  # Note: y-coord is horizontal (column)
            ys = [p[0] for p in path]  # Note: x-coord is vertical (row)
            
            # Plot path
            ax.plot(xs, ys, '-', color=colors[i % len(colors)], linewidth=2, 
                    alpha=0.7, label=f'Agent {i}')
            
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=colors[i % len(colors)])
            ax.plot(xs[-1], ys[-1], 's', color=colors[i % len(colors)])
        
        ax.set_title('Agent Paths Visualization')
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.legend()
        
        plt.savefig('agent_paths.png')
        plt.show()

    def analyze_q_values(self):
        """Analyze Q-values to understand agent decision making."""
        if not self.shared_q_table:
            print("No Q-table data available for analysis")
            return
        
        # Count states and analyze Q-value distributions
        state_count = len(self.shared_q_table)
        action_preferences = [0, 0, 0, 0]  # N, S, W, E
        
        # State type analysis
        states_with_neighbors = 0
        states_at_locations = {'A': 0, 'B': 0, 'other': 0}
        
        for state, q_values in self.shared_q_table.items():
            # Count preferred actions
            best_action = np.argmax(q_values)
            action_preferences[best_action] += 1
            
            # Count states with neighbors
            if state[3] != '0000':  # If any neighbor bits are set
                states_with_neighbors += 1
            
            # Count states at special locations
            if (state[0], state[1]) == self.a_location:
                states_at_locations['A'] += 1
            elif (state[0], state[1]) == self.b_location:
                states_at_locations['B'] += 1
            else:
                states_at_locations['other'] += 1
        
        # Print analysis results
        print("\nQ-Table Analysis:")
        print(f"Total states in Q-table: {state_count}")
        print(f"States with neighboring agents: {states_with_neighbors} ({states_with_neighbors/state_count*100:.2f}%)")
        print("\nStates by location:")
        for loc, count in states_at_locations.items():
            print(f"  {loc}: {count} ({count/state_count*100:.2f}%)")
        
        print("\nPreferred action distribution:")
        for i, count in enumerate(action_preferences):
            print(f"  {action_names[i]}: {count} ({count/state_count*100:.2f}%)")
        
        # Visualize action preferences
        plt.figure(figsize=(10, 6))
        plt.bar(action_names, action_preferences)
        plt.title('Preferred Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.savefig('action_preferences.png')
        plt.show()

def generate_test_configurations(num_configs=5, grid_size=GRID_SIZE):
    """Generate different A and B locations for testing."""
    configs = []
    
    # Always include the default config first
    configs.append((DEFAULT_A_LOCATION, DEFAULT_B_LOCATION))
    
    # Generate additional random configurations
    while len(configs) < num_configs:
        a_loc = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        b_loc = (random.randint