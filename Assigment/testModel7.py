import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pickle
import time

# Environment setup
GRID_SIZE = 5
NUM_AGENTS = 4
A_LOCATION = (1, 1)
B_LOCATION = (4, 4)

# Actions: 0=North, 1=South, 2=West, 3=East
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Agent:
    """
    Represents an agent in the grid world with a simple Q-table.
    Each agent learns to move from A to B and back while avoiding head-on collisions.
    Includes use of:
      - State of neighboring cells (only considers agents of opposite type)
      - Central clock (update schedule is round-robin)
      - Off-the-job training (start configs defined below)
    """
    def __init__(self, idx):
        self.idx = idx
        self.reset()
        self.q_table = {}  # Key: (x, y, carrying, neighbors), Value: [Q-values]
        self.epsilon = 0.2  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9999  # Decay rate for exploration
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.rewards_log = []
        self.deliveries = 0  # Track number of successful deliveries
        self.step_count = 0  # Track steps taken

    def reset(self):
        """Sets initial position and carrying status for off-the-job training."""
        self.pos = A_LOCATION if self.idx % 2 == 0 else B_LOCATION
        self.carrying = self.pos == A_LOCATION
        
    def get_neighbors_state(self, grid, agents):
        """
        Returns a binary string where 1 means an adjacent cell has an agent
        of opposite carrying state (e.g., one is carrying and the other is not).
        Improved to check diagonals as well (8-neighborhood).
        """
        # Check all 8 adjacent cells (N, NE, E, SE, S, SW, W, NW)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
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
        """Enhanced state representation including distance vectors to targets."""
        neighbors = self.get_neighbors_state(grid, agents)
        
        # Add distance to target (A if not carrying, B if carrying)
        target = B_LOCATION if self.carrying else A_LOCATION
        distance_x = target[0] - self.pos[0]
        distance_y = target[1] - self.pos[1]
        
        # Limit the range to keep the state space manageable
        distance_x = max(-2, min(2, distance_x))
        distance_y = max(-2, min(2, distance_y))
        
        return (self.pos[0], self.pos[1], int(self.carrying), neighbors, distance_x, distance_y)

    def choose_action(self, grid, agents, training=True):
        state = self.get_state(grid, agents)
        
        # During training, use epsilon-greedy with decaying epsilon
        if training and (random.random() < self.epsilon or state not in self.q_table):
            return random.randint(0, 3)
        
        # If state not in Q-table, initialize it with optimistic values
        if state not in self.q_table:
            self.q_table[state] = [0.1] * 4  # Slight optimism to encourage exploration
            
            # Add a slight bias toward the target
            target = B_LOCATION if self.carrying else A_LOCATION
            dx = target[0] - self.pos[0]
            dy = target[1] - self.pos[1]
            
            # Add small bias for actions toward target
            if dx > 0:  # Target is to the south
                self.q_table[state][1] += 0.05
            elif dx < 0:  # Target is to the north
                self.q_table[state][0] += 0.05
            if dy > 0:  # Target is to the east
                self.q_table[state][3] += 0.05
            elif dy < 0:  # Target is to the west
                self.q_table[state][2] += 0.05
                
        return int(np.argmax(self.q_table[state]))

    def update_q(self, prev_state, action, reward, next_state):
        if prev_state not in self.q_table:
            self.q_table[prev_state] = [0.0] * 4
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 4
            
        # Q-learning update rule
        max_next = max(self.q_table[next_state])
        self.q_table[prev_state][action] += self.alpha * (
            reward + self.gamma * max_next - self.q_table[prev_state][action]
        )
        
        # Decay exploration rate
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

# Initialize agents
agents = [Agent(i) for i in range(NUM_AGENTS)]

# Training settings
max_episodes = 10000
max_steps_per_episode = 25
step_budget = 1_500_000
collision_budget = 4000
walltime_budget = 600  # 10 minutes in seconds

def head_on_collision(agent, other):
    """Check if two agents have a head-on collision."""
    return agent.pos == other.pos and agent.carrying != other.carrying

# Improved training loop with monitoring
def train_agents():
    """Train the agents with a more structured approach."""
    start_time = time.time()
    total_steps = 0
    collision_count = 0
    episode = 0
    
    # For monitoring progress
    eval_interval = 500
    rewards_history = []
    collision_history = []
    success_rate_history = []
    
    print("Training agents with step/collision/time budget...")
    
    # Create a schedule for off-the-job training
    # We'll use different starting positions to help agents learn various scenarios
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
                
                # Get current state
                state = agent.get_state(grid, agents)
                
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
                
                # Check for collisions
                for other in agents:
                    if other.idx != agent.idx and head_on_collision(agent, other):
                        reward -= 10.0  # Strong penalty for collision
                        collision_count += 1
                        episode_collisions += 1
                
                # Small penalty for each step to encourage efficiency
                reward -= 0.01
                
                # Get new state
                next_state = agent.get_state(grid, agents)
                
                # Update Q-table
                agent.update_q(state, action, reward, next_state)
                
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
            print(f"Episode {episode}: Steps={total_steps}, Collisions={collision_count}, "
                  f"Time={elapsed_time:.1f}s, Success Rate={success_rate:.1f}%, "
                  f"Avg Reward={rewards_history[-1]:.2f}")
    
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
    plt.savefig('learning_curves.png')
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
                state = agent.get_state(grid, agents)
                
                # Choose action (no exploration during evaluation)
                if state in agent.q_table:
                    action = np.argmax(agent.q_table[state])
                else:
                    # If state not seen during training, use a simple heuristic
                    target = B_LOCATION if agent.carrying else A_LOCATION
                    dx = target[0] - agent.pos[0]
                    dy = target[1] - agent.pos[1]
                    
                    if abs(dx) > abs(dy):
                        action = 1 if dx > 0 else 0  # South or North
                    else:
                        action = 3 if dy > 0 else 2  # East or West
                
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
                    if other is not agent and head_on_collision(agent, other):
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

# Save trained Q-tables for future use
with open("q_tables.pkl", "wb") as f:
    pickle.dump([agent.q_table for agent in agents], f)

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
    Uses trained Q-tables for decision making.
    """
    draw_grid()
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for agent in agents:
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    for agent in agents:  # Round-robin scheduling
        state = agent.get_state(grid, agents)
        action = agent.choose_action(grid, agents, training=False)
        
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