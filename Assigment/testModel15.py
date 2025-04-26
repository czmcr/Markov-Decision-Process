"""
Enhanced Multi-Agent Q-Learning Script

This script trains multiple agents to shuttle items between two locations (A and B) on a grid,
avoiding head-on collisions. It logs detailed metrics, visualizes performance, conducts robustness
and collision analysis, and provides utilities for evaluation and model persistence.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pickle
import time
from collections import defaultdict, deque
import seaborn as sns

# ------------------------
# Environment Definitions
# ------------------------
GRID_SIZE = 5
NUM_AGENTS = 4
DEFAULT_A_LOCATION = (1, 1)
DEFAULT_B_LOCATION = (4, 4)

# Actions mapping
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_names = ["North", "South", "West", "East"]

# ------------------------
# Agent Class
# ------------------------
class Agent:
    """
    Tabular-Q Learning agent that shuttles items between A and B.
    Tracks recent positions to encourage path diversity during evaluation.
    """
    def __init__(self, idx, shared_q=None):
        self.idx = idx
        self.q_table = shared_q if shared_q is not None else {}
        self.epsilon = 0.2
        self.alpha = 1e-3
        self.gamma = 0.99
        self.reset()

    def reset(self, a_loc=DEFAULT_A_LOCATION, b_loc=DEFAULT_B_LOCATION):
        self.a_loc = a_loc
        self.b_loc = b_loc
        self.pos = a_loc if self.idx % 2 == 0 else b_loc
        self.carrying = (self.pos == a_loc)
        self.rewards_log = []
        self.last_visited = deque(maxlen=5)

    def reset_at_location(self, loc, carrying=False):
        self.pos = loc
        self.carrying = carrying
        self.last_visited.clear()

    def get_neighbors_state(self, grid, agents):
        bits = ''
        for dx, dy in actions:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                aid = grid[nx][ny]
                if aid != -1 and agents[aid].carrying != self.carrying:
                    bits += '1'
                    continue
            bits += '0'
        return bits

    def get_state(self, grid, agents):
        return (self.pos[0], self.pos[1], int(self.carrying), self.get_neighbors_state(grid, agents))

    def choose_action(self, grid, agents, eval_mode=False):
        state = self.get_state(grid, agents)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(4)
        q_values = np.array(self.q_table[state], dtype=float)
        if eval_mode:
            for a in range(4):
                dx, dy = actions[a]
                new_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if new_pos in self.last_visited:
                    q_values[a] -= 0.1
        return int(np.argmax(q_values))

    def update_q(self, s, a, r, s2):
        if s not in self.q_table:
            self.q_table[s] = [0.0] * 4
        if s2 not in self.q_table:
            self.q_table[s2] = [0.0] * 4
        td_error = r + self.gamma * max(self.q_table[s2]) - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error

    def move(self, a):
        dx, dy = actions[a]
        self.last_visited.append(self.pos)
        x, y = self.pos
        self.pos = (
            max(0, min(x + dx, GRID_SIZE - 1)),
            max(0, min(y + dy, GRID_SIZE - 1))
        )

    def update_carrying_status(self):
        if self.pos == self.a_loc and not self.carrying:
            self.carrying = True
            return 1
        if self.pos == self.b_loc and self.carrying:
            self.carrying = False
            return 1
        return 0

# ------------------------
# Environment Manager
# ------------------------
class Environment:
    def __init__(self):
        self.grid = [[-1] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.shared_q = {}
        self.agents = [Agent(i, self.shared_q) for i in range(NUM_AGENTS)]
        self.episodes = []
        self.avg_rewards = []
        self.collisions = []
        self.deliveries = []
        self.collision_map = defaultdict(int)
        self.success_window = deque(maxlen=100)
        self.step_budget = 1_500_000
        self.collision_budget = 4000
        self.time_budget = 600

    def head_on_collision(self, a, b):
        return a.pos == b.pos and a.carrying != b.carrying

    def reset(self, testing=False, a_loc=DEFAULT_A_LOCATION, b_loc=DEFAULT_B_LOCATION):
        self.ep_coll = 0
        self.ep_del = 0
        self.grid = [[-1] * GRID_SIZE for _ in range(GRID_SIZE)]
        for agent in self.agents:
            if testing:
                agent.reset_at_location(b_loc, carrying=False)
            else:
                agent.reset(a_loc, b_loc)
            self.grid[agent.pos[0]][agent.pos[1]] = agent.idx

    def train(self):
        start_time = time.time()
        total_steps = 0
        total_collisions = 0
        episode = 0
        print("Training start...")
        while (
            total_steps < self.step_budget and
            total_collisions < self.collision_budget and
            time.time() - start_time < self.time_budget
        ):
            self.reset()
            episode_reward = 0
            for _ in range(20):  # fixed-length episode
                for agent in self.agents:
                    state = agent.get_state(self.grid, self.agents)
                    action = agent.choose_action(self.grid, self.agents)
                    dx, dy = actions[action]
                    tx = max(0, min(agent.pos[0] + dx, GRID_SIZE - 1))
                    ty = max(0, min(agent.pos[1] + dy, GRID_SIZE - 1))
                    occ = self.grid[tx][ty]
                    if occ != -1 and self.agents[occ].carrying != agent.carrying:
                        reward = -1
                    else:
                        self.grid[agent.pos[0]][agent.pos[1]] = -1
                        agent.move(action)
                        reward = agent.update_carrying_status()
                        if reward > 0:
                            self.ep_del += 1
                        for other in self.agents:
                            if other is not agent and self.head_on_collision(agent, other):
                                reward = -10
                                self.ep_coll += 1
                                self.collision_map[agent.pos] += 1
                        self.grid[agent.pos[0]][agent.pos[1]] = agent.idx
                    agent.rewards_log.append(reward)
                    agent.update_q(state, action, reward,
                                  agent.get_state(self.grid, self.agents))
                    episode_reward += reward
                    total_steps += 1
            total_collisions += self.ep_coll
            self.episodes.append(episode)
            self.avg_rewards.append(episode_reward / NUM_AGENTS)
            self.collisions.append(self.ep_coll)
            self.deliveries.append(self.ep_del)
            self.success_window.append(1 if self.ep_del > 0 and self.ep_coll == 0 else 0)
            if episode % 100 == 0:
                success_rate = np.mean(self.success_window) * 100
                print(f"Ep {episode}: R={self.avg_rewards[-1]:.2f}, "
                      f"D={self.ep_del}, C={self.ep_coll}, Success~{success_rate:.1f}%")
            episode += 1
        elapsed = time.time() - start_time
        print(f"Training done: Steps={total_steps}, Collisions={total_collisions}, Time={elapsed:.2f}s")

    def evaluate(self, trials=100, max_steps=25):
        success_count = 0
        collision_count = 0
        delivery_steps = []
        for _ in range(trials):
            self.reset(testing=True)
            collided = False
            first_delivery = None
            for step in range(max_steps):
                for agent in self.agents:
                    action = agent.choose_action(self.grid, self.agents, eval_mode=True)
                    self.grid[agent.pos[0]][agent.pos[1]] = -1
                    agent.move(action)
                    agent.update_carrying_status()
                    self.grid[agent.pos[0]][agent.pos[1]] = agent.idx
                for a1 in self.agents:
                    for a2 in self.agents:
                        if a1 is not a2 and self.head_on_collision(a1, a2):
                            collided = True
                if any(agent.pos == DEFAULT_B_LOCATION and not agent.carrying
                       for agent in self.agents):
                    first_delivery = step
                    break
                if collided:
                    break
            if first_delivery is not None and not collided:
                success_count += 1
                delivery_steps.append(first_delivery)
            if collided:
                collision_count += 1
        success_rate = success_count / trials * 100
        collision_rate = collision_count / trials * 100
        avg_step = np.mean(delivery_steps) if delivery_steps else float('nan')
        print(f"Eval: Success={success_rate:.1f}%, Collision={collision_rate:.1f}%, "
              f"AvgStep={avg_step:.2f}")
        return {
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_step': avg_step
        }

    def visualize(self):
        plt.figure()
        plt.plot(self.episodes, self.avg_rewards)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')
        plt.show()

        plt.figure()
        plt.plot(self.episodes, self.collisions)
        plt.title('Collisions per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Collisions')
        plt.show()

        heat = np.zeros((GRID_SIZE, GRID_SIZE))
        for (x, y), count in self.collision_map.items():
            heat[x, y] = count
        plt.figure()
        sns.heatmap(heat, annot=True, cmap='YlOrRd')
        plt.title('Collision Heatmap')
        plt.show()

    def save(self, fname='q_tables.pkl'):
        """Save trained Q-table to a file."""
        with open(fname, 'wb') as file:
            pickle.dump(self.shared_q, file)
        print(f"Saved model to {fname}")

    def load(self, fname='q_tables.pkl'):
        """Load a trained Q-table from a file."""
        try:
            with open(fname, 'rb') as file:
                data = pickle.load(file)
                self.shared_q.update(data)
                for agent in self.agents:
                    agent.q_table = self.shared_q
            print(f"Loaded model from {fname}")
        except FileNotFoundError:
            print(f"Model file {fname} not found.")

if __name__ == '__main__':
    env = Environment()
    env.train()
    env.evaluate()
    env.visualize()
    env.save()
