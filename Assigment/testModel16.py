"""
Enhanced Multi-Agent Q-Learning with Animated Visualization

This script trains multiple agents to shuttle items between two locations (A and B) on a grid
avoiding head-on collisions. It includes live animations of agent movements, performance
metrics, collision analysis, evaluation, and persistence of the learned Q-table.
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

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_names = ["North", "South", "West", "East"]

# ------------------------
# Agent Class
# ------------------------
class Agent:
    """
    Tabular-Q Learning agent that shuttles between A and B.
    Records recent positions to encourage path diversity.
    """
    def __init__(self, idx, shared_q=None):
        self.idx = idx
        self.q_table = shared_q if shared_q is not None else {}
        self.epsilon, self.alpha, self.gamma = 0.2, 1e-3, 0.99
        self.reset()

    def reset(self, a_loc=DEFAULT_A_LOCATION, b_loc=DEFAULT_B_LOCATION):
        self.a_loc, self.b_loc = a_loc, b_loc
        self.pos = a_loc if self.idx % 2 == 0 else b_loc
        self.carrying = (self.pos == a_loc)
        self.rewards_log = []
        self.last_visited = deque(maxlen=5)

    def reset_at_location(self, loc, carrying=False):
        self.pos, self.carrying = loc, carrying
        self.last_visited.clear()

    def get_neighbors_state(self, grid, agents):
        bits = ''
        for dx, dy in actions:
            nx, ny = self.pos[0]+dx, self.pos[1]+dy
            if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE and grid[nx][ny]!=-1:
                other = agents[grid[nx][ny]]
                bits += '1' if other.carrying != self.carrying else '0'
            else:
                bits += '0'
        return bits

    def get_state(self, grid, agents):
        return (self.pos[0], self.pos[1], int(self.carrying), self.get_neighbors_state(grid, agents))

    def choose_action(self, grid, agents, eval_mode=False):
        s = self.get_state(grid, agents)
        if s not in self.q_table: self.q_table[s] = [0.0]*4
        if not eval_mode and random.random()<self.epsilon:
            return random.randrange(4)
        q_vals = np.array(self.q_table[s], dtype=float)
        if eval_mode:
            for a in range(4):
                dx, dy = actions[a]
                if (self.pos[0]+dx, self.pos[1]+dy) in self.last_visited:
                    q_vals[a] -= 0.1
        return int(np.argmax(q_vals))

    def update_q(self, s, a, r, s2):
        for state in (s, s2):
            if state not in self.q_table: self.q_table[state]=[0.0]*4
        td = r + self.gamma*max(self.q_table[s2]) - self.q_table[s][a]
        self.q_table[s][a] += self.alpha*td

    def move(self, a):
        dx, dy = actions[a]
        self.last_visited.append(self.pos)
        x,y = self.pos
        self.pos = (max(0,min(x+dx,GRID_SIZE-1)), max(0,min(y+dy,GRID_SIZE-1)))

    def update_carrying_status(self):
        if self.pos==self.a_loc and not self.carrying:
            self.carrying=True; return 1
        if self.pos==self.b_loc and self.carrying:
            self.carrying=False; return 1
        return 0

# ------------------------
# Environment Manager
# ------------------------
class Environment:
    """
    Manages agents, training, evaluation, persistence, and animated visualization.
    """
    def __init__(self):
        self.grid=[[ -1]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.shared_q={}
        self.agents=[Agent(i,self.shared_q) for i in range(NUM_AGENTS)]
        self.episodes, self.avg_rewards, self.collisions, self.deliveries = [],[],[],[]
        self.collision_map=defaultdict(int)
        self.success_window=deque(maxlen=100)
        self.step_budget, self.collision_budget, self.time_budget = 1_500_000,4000,600

    def head_on_collision(self,a,b): return a.pos==b.pos and a.carrying!=b.carrying

    def reset(self,testing=False, a_loc=DEFAULT_A_LOCATION, b_loc=DEFAULT_B_LOCATION):
        self.ep_coll=self.ep_del=0
        self.grid=[[ -1]*GRID_SIZE for _ in range(GRID_SIZE)]
        for ag in self.agents:
            if testing: ag.reset_at_location(b_loc,False)
            else: ag.reset(a_loc,b_loc)
            self.grid[ag.pos[0]][ag.pos[1]]=ag.idx

    def train(self):
        start=time.time(); steps=0; col=0; ep=0
        while steps<self.step_budget and col<self.collision_budget and time.time()-start<self.time_budget:
            self.reset()
            ep_reward=0
            for _ in range(20):
                for ag in self.agents:
                    s=ag.get_state(self.grid,self.agents)
                    a=ag.choose_action(self.grid,self.agents)
                    dx,dy=actions[a]
                    tx,ty=max(0,min(ag.pos[0]+dx,GRID_SIZE-1)),max(0,min(ag.pos[1]+dy,GRID_SIZE-1))
                    if self.grid[tx][ty]!=-1 and self.agents[self.grid[tx][ty]].carrying!=ag.carrying:
                        r=-1
                    else:
                        self.grid[ag.pos[0]][ag.pos[1]]=-1
                        ag.move(a)
                        r=ag.update_carrying_status()
                        if r>0: self.ep_del+=1
                        for other in self.agents:
                            if other!=ag and self.head_on_collision(ag,other):
                                r=-10; self.ep_coll+=1; self.collision_map[ag.pos]+=1
                        self.grid[ag.pos[0]][ag.pos[1]]=ag.idx
                    ag.update_q(s,a,r,ag.get_state(self.grid,self.agents))
                    ep_reward+=r; steps+=1
            col+=self.ep_coll
            self.episodes.append(ep); self.avg_rewards.append(ep_reward/NUM_AGENTS)
            self.collisions.append(self.ep_coll); self.deliveries.append(self.ep_del)
            self.success_window.append(1 if self.ep_del>0 and self.ep_coll==0 else 0)
            if ep%100==0:
                print(f"Ep {ep}: R={self.avg_rewards[-1]:.2f}, D={self.ep_del}, C={self.ep_coll}, S~{np.mean(self.success_window)*100:.1f}%")
            ep+=1
        print("Training complete",steps,col,time.time()-start)

    def evaluate(self,trials=100,max_steps=25):
        sc=cc=0; ds=[]
        for _ in range(trials):
            self.reset(testing=True); collided=False; first=None
            for t in range(max_steps):
                for ag in self.agents:
                    a=ag.choose_action(self.grid,self.agents,eval_mode=True)
                    self.grid[ag.pos[0]][ag.pos[1]]=-1; ag.move(a); ag.update_carrying_status(); self.grid[ag.pos[0]][ag.pos[1]]=ag.idx
                if any(ag.pos==DEFAULT_B_LOCATION and not ag.carrying for ag in self.agents): first=t; break
                if any(self.head_on_collision(a1,a2) for a1 in self.agents for a2 in self.agents if a1!=a2): collided=True; break
            if first is not None and not collided: sc+=1; ds.append(first)
            if collided: cc+=1
        print(f"Eval: S={sc}/{trials}, C={cc}, AvgStep={np.mean(ds) if ds else float('nan')}")

    def visualize(self):
        # static plots
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        axes[0].plot(self.episodes,self.avg_rewards); axes[0].set_title('Avg Reward')
        axes[1].plot(self.episodes,self.collisions); axes[1].set_title('Collisions')
        plt.show()
        heat=np.zeros((GRID_SIZE,GRID_SIZE))
        for (x,y),c in self.collision_map.items(): heat[x,y]=c
        plt.figure(); sns.heatmap(heat,annot=True,cmap='YlOrRd'); plt.title('Collision Heatmap'); plt.show()

    def animate(self, frames=100, interval=300):
        # Initialize for animation
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0.5, GRID_SIZE, 1)); ax.set_yticks(np.arange(0.5, GRID_SIZE, 1))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True); ax.invert_yaxis()
        ax.text(DEFAULT_A_LOCATION[1],DEFAULT_A_LOCATION[0],'A',color='green',ha='center')
        ax.text(DEFAULT_B_LOCATION[1],DEFAULT_B_LOCATION[0],'B',color='blue',ha='center')
        scatters = [ax.plot([], [], 'o', markersize=12)[0] for _ in range(NUM_AGENTS)]
        texts = [ax.text(0,0,'') for _ in range(NUM_AGENTS)]

        # Reset agents at B for visualization
        self.reset(testing=True)

        def update(frame):
            # Step all agents once
            for i, ag in enumerate(self.agents):
                self.grid[ag.pos[0]][ag.pos[1]]=-1
                a = ag.choose_action(self.grid,self.agents,eval_mode=True)
                ag.move(a); ag.update_carrying_status()
                self.grid[ag.pos[0]][ag.pos[1]] = ag.idx
                # update scatter
                color = 'green' if ag.carrying else 'blue'
                scatters[i].set_data([ag.pos[1]], [ag.pos[0]]); scatters[i].set_color(color)
                texts[i].set_position((ag.pos[1], ag.pos[0]+0.3))
                texts[i].set_text(f"{i}")
            return scatters + texts

        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.show()

    def save(self, fname='q_tables.pkl'):
        with open(fname,'wb') as f: pickle.dump(self.shared_q,f)
        print(f"Saved model to {fname}")

    def load(self, fname='q_tables.pkl'):
        try:
            with open(fname,'rb') as f: data=pickle.load(f)
            self.shared_q.update(data)
            for ag in self.agents: ag.q_table=self.shared_q
            print(f"Loaded model from {fname}")
        except FileNotFoundError:
            print(f"Model file {fname} not found.")

if __name__=='__main__':
    env=Environment()
    env.train()
    env.evaluate()
    env.visualize()
    env.animate(frames=200, interval=300)
    env.save()
