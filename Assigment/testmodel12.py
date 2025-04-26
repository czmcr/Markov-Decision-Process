# Multi-Agent Q-Learning Environment for FIT5226 Project

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
    def __init__(self, idx, shared_q=None):
        self.idx = idx
        self.reset()
        # Use shared Q-table if provided, otherwise create individual Q-table
        self.q_table = shared_q if shared_q is not None else {}
        self.epsilon = 0.2
        self.alpha = 1e-3
        self.gamma = 0.99
        self.rewards_log = []

    def reset(self):
        """Sets initial position and carrying status for off-the-job training."""
        self.pos = A_LOCATION if self.idx % 2 == 0 else B_LOCATION
        self.carrying = self.pos == A_LOCATION

    def reset_at_b(self):
        """Reset agent to start at B for performance evaluation."""
        self.pos = B_LOCATION
        self.carrying = False  # Not carrying when starting at B

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
        if not eval_mode and random.random() < self.epsilon or state not in self.q_table:
            return random.randint(0, 3)
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
        self.pos = (new_x, new_y)

    def update_carrying_status(self):
        """Updates carrying status based on current position"""
        # When at A, pickup supply (become carrying)
        if self.pos == A_LOCATION and not self.carrying:
            self.carrying = True
            return 1  # Reward for pickup
        # When at B, deliver supply (become not carrying)
        elif self.pos == B_LOCATION and self.carrying:
            self.carrying = False
            return 1  # Reward for delivery
        return 0  # No reward if no pickup/delivery happened

# Create a shared Q-table for all agents
shared_q_table = {}

# Initialize agents with shared Q-table
agents = [Agent(i, shared_q=shared_q_table) for i in range(NUM_AGENTS)]

# Training pipeline constraints
step_budget = 1_500_000
collision_budget = 4000
walltime_budget = 600  # 10 minutes in seconds
start_time = time.time()
total_steps = 0
collision_count = 0

def head_on_collision(agent, other):
    return agent.pos == other.pos and agent.carrying != other.carrying

# Training loop with budgets
print("Training agents with step/collision/time budget...")
episode = 0
while total_steps < step_budget and collision_count < collision_budget and (time.time() - start_time) < walltime_budget:
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for agent in agents:
        agent.reset()
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    for step in range(20):
        for agent in agents:
            total_steps += 1
            state = agent.get_state(grid, agents)
            action = agent.choose_action(grid, agents)
            dx, dy = actions[action]
            proposed_x = min(max(agent.pos[0] + dx, 0), GRID_SIZE - 1)
            proposed_y = min(max(agent.pos[1] + dy, 0), GRID_SIZE - 1)

            # Allow move only if the target cell is empty OR occupied by agents going in the same direction
            occupying_agent_id = grid[proposed_x][proposed_y]
            if occupying_agent_id != -1:
                occupying_agent = agents[occupying_agent_id]
                if occupying_agent.carrying != agent.carrying:
                    reward = -1  # discourage illegal move into conflicting traffic
                    agent.rewards_log.append(reward)
                    continue

            # Clear the agent's position in the grid
            grid[agent.pos[0]][agent.pos[1]] = -1
            old_pos = agent.pos
            agent.move(action)

            # Initialize reward
            reward = 0
            
            # Add reward for completing pickup or delivery
            reward += agent.update_carrying_status()
            
            # Check for collisions
            for other in agents:
                if other is not agent and head_on_collision(agent, other):
                    reward = -10
                    collision_count += 1

            agent.rewards_log.append(reward)
            next_state = agent.get_state(grid, agents)
            agent.update_q(state, action, reward, next_state)
            
            # Update the agent's position in the grid
            grid[agent.pos[0]][agent.pos[1]] = agent.idx

    episode += 1

print(f"Training completed: Episodes={episode}, Steps={total_steps}, Collisions={collision_count}, Time Elapsed={time.time() - start_time:.2f}s")

# Save trained Q-tables for future use
with open("q_tables.pkl", "wb") as f:
    pickle.dump([shared_q_table], f)  # Save only the shared Q-table

# Performance Evaluation Function
def evaluate_performance(num_trials=100, max_steps=25):
    """
    Evaluates agent performance starting at B.
    Success criteria: Complete delivery (B→A→B) within max_steps steps without collisions.
    Returns success rate.
    """
    print(f"\nEvaluating final performance over {num_trials} trials ({max_steps} steps max)...")
    
    successful_trials = 0
    
    for trial in range(num_trials):
        # Reset environment
        grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        # Reset all agents to start at B
        for agent in agents:
            agent.reset_at_b()
            grid[agent.pos[0]][agent.pos[1]] = agent.idx
        
        # Track if this trial had any collisions
        collision_occurred = False
        # Track if agents completed a delivery in this trial
        delivery_completed = [False] * NUM_AGENTS
        
        # Run the trial for max_steps steps
        for step in range(max_steps):
            for agent_idx, agent in enumerate(agents):
                # Get state and choose action (no exploration during evaluation)
                state = agent.get_state(grid, agents)
                action = agent.choose_action(grid, agents, eval_mode=True)
                
                # Clear agent's current position in grid
                grid[agent.pos[0]][agent.pos[1]] = -1
                
                # Execute move
                agent.move(action)
                
                # Check for delivery completion: B→A→B cycle
                if agent.pos == A_LOCATION and not agent.carrying:
                    agent.carrying = True  # Pick up at A
                elif agent.pos == B_LOCATION and agent.carrying:
                    agent.carrying = False  # Deliver at B
                    delivery_completed[agent_idx] = True
                
                # Check for collisions
                for other in agents:
                    if other is not agent and head_on_collision(agent, other):
                        collision_occurred = True
                
                # Update grid with new position
                grid[agent.pos[0]][agent.pos[1]] = agent.idx
            
            # If all agents completed delivery or collision occurred, end trial
            if all(delivery_completed) or collision_occurred:
                break
        
        # Trial is successful if at least one agent completed delivery and no collisions
        if any(delivery_completed) and not collision_occurred:
            successful_trials += 1
    
    success_rate = successful_trials / num_trials * 100
    print(f"Performance results: {successful_trials}/{num_trials} successful trials ({success_rate:.2f}%)")
    
    # Check if performance meets requirement (75% success rate)
    if success_rate >= 75:
        print("Performance requirement met (≥75% success rate)")
    else:
        print("Performance requirement not met (<75% success rate)")
    
    return success_rate

# Run evaluation after training
evaluation_result = evaluate_performance()

# Visualization setup
fig, ax = plt.subplots()

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
    ax.set_title(f"Multi-Agent Delivery: A ↔ B (Success Rate: {evaluation_result:.1f}%)", fontsize=14)

def update(frame):
    """
    Updates the grid visualization and performs one Q-learning step for each agent.
    Agents are updated in round-robin order (central clock option).
    Uses a grid to track positions for state sensing.
    """
    draw_grid()
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for agent in agents:
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    for agent in agents:  # Round-robin scheduling
        state = agent.get_state(grid, agents)
        action = agent.choose_action(grid, agents, eval_mode=True)  # No exploration during visualization
        
        # Clear the agent's position in the grid
        grid[agent.pos[0]][agent.pos[1]] = -1
        
        old_pos = agent.pos
        agent.move(action)
        
        # Initialize reward
        reward = 0
        
        # Add reward for completing pickup or delivery
        reward += agent.update_carrying_status()

        # Check for collisions
        for other in agents:
            if other is not agent and head_on_collision(agent, other):
                reward = -10  # Penalize head-on collision

        agent.rewards_log.append(reward)
        
        # Update the agent's position in the grid
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    # Plot agent positions
    for agent in agents:
        color = 'green' if agent.carrying else 'blue'
        label = f"A{agent.idx} {'[Carrying]' if agent.carrying else '[Empty]'}"
        ax.plot(agent.pos[1], agent.pos[0], 'o', markersize=12, color=color)
        ax.text(agent.pos[1], agent.pos[0], f"{agent.idx}", ha='center', va='center', color='white', fontsize=8)
        ax.text(agent.pos[1], agent.pos[0] + 0.3, label, ha='center', va='bottom', fontsize=6)

    fig.canvas.draw()

# Initialize agents at B for the visualization
for agent in agents:
    agent.reset_at_b()

ani = FuncAnimation(fig, update, frames=200, interval=300, repeat=False)
plt.show()