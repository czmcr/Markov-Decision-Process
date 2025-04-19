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
A_LOCATION = (0, 0)
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
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99
        self.rewards_log = []

    def reset(self):
        """Sets initial position and carrying status for off-the-job training."""
        self.pos = A_LOCATION if self.idx % 2 == 0 else B_LOCATION
        self.carrying = self.pos == A_LOCATION

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

    def choose_action(self, grid, agents):
        state = self.get_state(grid, agents)
        if random.random() < self.epsilon or state not in self.q_table:
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
        if self.pos == A_LOCATION:
            self.carrying = True
        elif self.pos == B_LOCATION:
            self.carrying = False

# Initialize agents
agents = [Agent(i) for i in range(NUM_AGENTS)]

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

            # Apply penalty if the target cell is occupied by an agent going in the opposite direction
            occupying_agent_id = grid[proposed_x][proposed_y]
            if occupying_agent_id != -1:
                occupying_agent = agents[occupying_agent_id]
                if occupying_agent.carrying != agent.carrying:
                    reward = -5  # apply penalty but allow move

            old_pos = agent.pos
            agent.move(action)

            reward = 1 if (old_pos == A_LOCATION and agent.pos == B_LOCATION) else 0
            for other in agents:
                if other is not agent and head_on_collision(agent, other):
                    reward = -10
                    collision_count += 1

            agent.rewards_log.append(reward)
            next_state = agent.get_state(grid, agents)
            agent.update_q(state, action, reward, next_state)
            grid[agent.pos[0]][agent.pos[1]] = agent.idx

    episode += 1

print(f"Training completed: Episodes={episode}, Steps={total_steps}, Collisions={collision_count}, Time Elapsed={time.time() - start_time:.2f}s")

# Evaluate performance after training
successful_deliveries = 0
collision_free_runs = 0
total_evaluation_runs = 100
max_deliveries = total_evaluation_runs * NUM_AGENTS
bonus_qualified_episodes = 0

for _ in range(total_evaluation_runs):
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for agent in agents:
        agent.reset()
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    delivery_success = [False] * NUM_AGENTS
    local_collisions = 0
    steps_taken = 0

    for step in range(25):
        steps_taken += 1
        for agent in agents:
            state = agent.get_state(grid, agents)
            action = np.argmax(agent.q_table[state]) if state in agent.q_table else random.randint(0, 3)
            dx, dy = actions[action]
            proposed_x = min(max(agent.pos[0] + dx, 0), GRID_SIZE - 1)
            proposed_y = min(max(agent.pos[1] + dy, 0), GRID_SIZE - 1)

            occupying_agent_id = grid[proposed_x][proposed_y]
            if occupying_agent_id != -1:
                occupying_agent = agents[occupying_agent_id]
                if occupying_agent.carrying != agent.carrying:
                    local_collisions += 1

            agent.move(action)
            if agent.pos == B_LOCATION:
                delivery_success[agent.idx] = True
            grid[agent.pos[0]][agent.pos[1]] = agent.idx

    successful_deliveries += sum(delivery_success)
    if all(delivery_success) and steps_taken <= 20 and local_collisions == 0:
        bonus_qualified_episodes += 1
    if local_collisions == 0:
        collision_free_runs += 1

success_rate = (successful_deliveries / max_deliveries) * 100
collision_free_rate = (collision_free_runs / total_evaluation_runs) * 100
bonus_success_rate = (bonus_qualified_episodes / total_evaluation_runs) * 100

print("\n--- Evaluation Results ---")
print(f"Success Rate (individual deliveries in ≤25 steps): {success_rate:.2f}%")
print(f"Collision-Free Runs: {collision_free_rate:.2f}%")
print(f"Bonus Evaluation Success Rate (<20 steps & no collisions): {bonus_success_rate:.2f}%")

# Save trained Q-tables for future use
with open("q_tables.pkl", "wb") as f:
    pickle.dump([agent.q_table for agent in agents], f)
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
    ax.set_title("Multi-Agent Delivery: A ↔ B", fontsize=14)

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
        action = agent.choose_action(grid, agents)
        old_pos = agent.pos
        agent.move(action)
        reward = 1 if (old_pos == A_LOCATION and agent.pos == B_LOCATION) else 0

        for other in agents:
            if other is not agent and head_on_collision(agent, other):
                reward = -10  # Penalize head-on collision

        agent.rewards_log.append(reward)
        next_state = agent.get_state(grid, agents)
        agent.update_q(state, action, reward, next_state)
        grid[agent.pos[0]][agent.pos[1]] = agent.idx

    # Plot agent positions
    for agent in agents:
        color = 'green' if agent.carrying else 'blue'
        label = f"A{agent.idx} {'[Carrying]' if agent.carrying else '[Empty]'}"
        ax.plot(agent.pos[1], agent.pos[0], 'o', markersize=12, color=color)
        ax.text(agent.pos[1], agent.pos[0], f"{agent.idx}", ha='center', va='center', color='white', fontsize=8)
        ax.text(agent.pos[1], agent.pos[0] + 0.3, label, ha='center', va='bottom', fontsize=6)

    fig.canvas.draw()

ani = FuncAnimation(fig, update, frames=200, interval=300, repeat=False)
plt.show()
