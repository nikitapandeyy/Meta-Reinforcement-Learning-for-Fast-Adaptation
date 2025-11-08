# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random

st.set_page_config(page_title="Meta-RL Maze Demo", layout="wide")

# ----------------------
# Helper: Maze & Q-utils
# ----------------------
class Maze:
    def __init__(self, maze_map=None, size=5, start=(0,0), goal=None):
        if maze_map is None:
            self.map = np.zeros((size, size), dtype=int)
        else:
            self.map = np.array(maze_map, dtype=int)
        self.size = self.map.shape[0]
        self.start = start
        self.goal = goal if goal is not None else (self.size-1, self.size-1)

    def is_free(self, pos):
        y,x = pos
        if 0 <= y < self.size and 0 <= x < self.size:
            return self.map[y,x] == 0
        return False

    def render_grid(self):
        grid = np.copy(self.map).astype(float)
        gy,gx = self.goal
        sy,sx = self.start
        grid[gy,gx] = 0.5   # goal highlight
        grid[sy,sx] = 0.8   # start highlight
        return grid

actions = ['up','down','left','right']

def next_state(state, action, maze: Maze):
    y,x = state
    if action == 'up':    y -= 1
    elif action == 'down': y += 1
    elif action == 'left': x -= 1
    elif action == 'right': x += 1
    if maze.is_free((y,x)):
        return (y,x)
    return state

def default_reward(new_state, goal):
    return 10 if new_state == goal else -1

# ----------------------
# Q-learning functions
# ----------------------
def init_Q(size):
    return np.zeros((size,size, len(actions)))

def q_learning_train(maze: Maze, episodes=200, alpha=0.7, gamma=0.9, eps=0.1, max_steps=100):
    Q = init_Q(maze.size)
    rewards = []
    for ep in range(episodes):
        state = maze.start
        total = 0
        for step in range(max_steps):
            if random.random() < eps:
                a_idx = random.randint(0, len(actions)-1)
            else:
                a_idx = int(np.argmax(Q[state[0], state[1]]))
            a = actions[a_idx]
            new_s = next_state(state, a, maze)
            r = default_reward(new_s, maze.goal)
            total += r
            Q[state[0], state[1], a_idx] += alpha * (r + gamma * np.max(Q[new_s[0], new_s[1]]) - Q[state[0], state[1], a_idx])
            state = new_s
            if state == maze.goal:
                break
        rewards.append(total)
    return Q, rewards

def extract_policy_path(Q, maze: Maze, max_steps=50):
    path = [maze.start]
    state = maze.start
    for _ in range(max_steps):
        a_idx = int(np.argmax(Q[state[0], state[1]]))
        state = next_state(state, actions[a_idx], maze)
        path.append(state)
        if state == maze.goal:
            break
        if len(path) > max_steps:
            break
    return path

# ----------------------
# Simple meta-training:
# Train short Q on many tasks, average Q-tables -> Q_meta
# (This is a demo-friendly meta-init approximation)
# ----------------------
def meta_train_aggregate(task_mazes, per_task_episodes=100, **q_kwargs):
    qs = []
    task_rewards = []
    for m in task_mazes:
        Q_t, rewards = q_learning_train(m, episodes=per_task_episodes, **q_kwargs)
        qs.append(Q_t)
        task_rewards.append(rewards)
    Q_meta = np.mean(qs, axis=0)
    return Q_meta, task_rewards

# ----------------------
# UI layout
# ----------------------
st.title("Meta-RL Maze Demo (Colab-friendly)")
st.markdown('''
**What this demo shows:**
- A simple *meta-learning* demo using Q-learning as a base: we train short Q-learners on many related mazes, average their Q-tables as a meta-init, then fine-tune on a new unseen maze.
- Compare **Meta-init + Adaptation** vs **From-scratch Q-learning** and visualize step-by-step agent paths.
''')

col1, col2 = st.columns([1,2])

with col1:
    st.header("Settings")
    size = st.slider("Maze size", 4, 8, 5)
    n_meta_tasks = st.slider("Number of meta-training mazes", 2, 6, 4)
    meta_epochs = st.slider("Per-task (meta) short train episodes", 20, 300, 80)
    adapt_epochs = st.slider("Adaptation episodes (fine-tune)", 10, 300, 100)
    baseline_epochs = st.slider("Baseline (scratch) episodes", 10, 500, 200)
    seed = st.number_input("Random seed (for reproducibility)", value=42)
    random.seed(seed); np.random.seed(seed)

    st.subheader("Buttons")
    b_meta = st.button("Run Meta-Train (short tasks)")
    b_adapt = st.button("Adapt on New Maze (show step-by-step)")
    b_baseline = st.button("Train Baseline from Scratch (show step-by-step)")
    st.write("Note: meta-train runs faster per task; adaptation visual shows step animation.")

with col2:
    st.header("Visualization")
    vis_placeholder = st.empty()
    chart_placeholder = st.empty()

# ----------------------
# Prepare meta-task mazes
# ----------------------
def random_maze(size, wall_prob=0.15):
    # random walls but keep start and goal free
    m = np.zeros((size,size), dtype=int)
    for i in range(size):
        for j in range(size):
            if random.random() < wall_prob:
                m[i,j] = 1
    m[0,0] = 0
    m[size-1,size-1] = 0
    return m

meta_mazes = []
for i in range(n_meta_tasks):
    mm = random_maze(size, wall_prob=0.18)
    # ensure there is at least a naive open path by clearing a diagonal
    for d in range(size):
        mm[d,d] = 0
    meta_mazes.append(Maze(maze_map=mm, size=size, start=(0,0), goal=(size-1,size-1)))

# New unseen test maze (shifted goal)
test_map = random_maze(size, wall_prob=0.18)
# ensure path possibility
for d in range(size):
    test_map[d, size-1-d] = 0
test_maze = Maze(maze_map=test_map, size=size, start=(0,0), goal=(size-1,size-1))

# ----------------------
# Meta-Train
# ----------------------
if b_meta:
    st.info("Running meta-training across short tasks (this may take a moment)...")
    with st.spinner("Meta-training..."):
        meta_q, per_task_rewards = meta_train_aggregate(meta_mazes, per_task_episodes=meta_epochs, alpha=0.7, gamma=0.9, eps=0.15)
    st.success("Meta-training done. Meta-initialization Q_meta computed.")
    # show a quick chart of per-task rewards (averaged)
    avg_task_rewards = np.mean(np.array([np.mean(r) for r in per_task_rewards]))
    st.write(f"Average per-task mean reward (meta tasks): {avg_task_rewards:.2f}")
    # store in session state
    st.session_state['Q_meta'] = meta_q

# ----------------------
# Adaptation using meta-init
# ----------------------
def animate_policy_run(Q_init, maze: Maze, adapt_episodes=100, show_live=True, title_prefix=""):
    # Fine-tune starting from Q_init
    Q = np.copy(Q_init)
    rewards = []
    for ep in range(adapt_episodes):
        state = maze.start
        total = 0
        for step in range(200):
            if random.random() < 0.05:
                a_idx = random.randint(0, len(actions)-1)
            else:
                a_idx = int(np.argmax(Q[state[0], state[1]]))
            a = actions[a_idx]
            new_s = next_state(state, a, maze)
            r = default_reward(new_s, maze.goal)
            total += r
            # small learning rate for adaptation
            Q[state[0], state[1], a_idx] += 0.5 * (r + 0.9 * np.max(Q[new_s[0], new_s[1]]) - Q[state[0], state[1], a_idx])
            state = new_s
            if state == maze.goal:
                break
        rewards.append(total)
        # show intermediate path every few episodes
        if show_live and (ep % max(1, adapt_episodes//8) == 0 or ep == adapt_episodes-1):
            path = extract_policy_path(Q, maze, max_steps=maze.size*4)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(maze.render_grid(), cmap='Greys', origin='upper')
            xs = [p[1] for p in path]; ys = [p[0] for p in path]
            if len(path)>0:
                ax.plot(xs, ys, marker='o', color='red')
            ax.scatter([maze.start[1]],[maze.start[0]], color='green', s=100, label='Start')
            ax.scatter([maze.goal[1]],[maze.goal[0]], color='gold', s=100, label='Goal')
            ax.set_title(f"{title_prefix} Adapt Ep {ep+1}/{adapt_episodes}")
            ax.set_xticks(range(maze.size)); ax.set_yticks(range(maze.size))
            ax.grid(True)
            vis_placeholder.pyplot(fig)
            time.sleep(0.3)
    return Q, rewards

if b_adapt:
    # require meta trained Q_meta
    if 'Q_meta' not in st.session_state:
        st.warning("Meta-init not found. Please press 'Run Meta-Train' first to create Q_meta.")
    else:
        st.info("Adapting meta-init to new maze (step-by-step)...")
        Qmeta = st.session_state['Q_meta']
        Q_after, adapt_rewards = animate_policy_run(Qmeta, test_maze, adapt_episodes=adapt_epochs, show_live=True, title_prefix="Meta-init")
        st.success("Adaptation from meta-init completed.")
        # show reward curve
        chart_placeholder.line_chart(np.array(adapt_rewards).cumsum())

if b_baseline:
    st.info("Training baseline Q-learning from scratch (step-by-step)...")
    # start from zeros
    Q0 = init_Q(test_maze.size)
    Q_after_baseline, baseline_rewards = animate_policy_run(Q0, test_maze, adapt_episodes=baseline_epochs, show_live=True, title_prefix="Baseline")
    st.success("Baseline training completed.")
    chart_placeholder.line_chart(np.array(baseline_rewards).cumsum())

# ----------------------
# Always show test maze static preview and meta tasks preview
# ----------------------
with st.expander("Show test maze and meta-task examples"):
    cols = st.columns(len(meta_mazes)+1)
    for i, m in enumerate(meta_mazes):
        ax = cols[i].pyplot(plt.figure(figsize=(2,2)))
        fig, a = plt.subplots(figsize=(2,2)); a.imshow(m.render_grid(), cmap='Greys', origin='upper'); a.set_title(f"Meta {i+1}"); a.set_xticks([]); a.set_yticks([])
        cols[i].pyplot(fig)
    # test maze
    fig, a = plt.subplots(figsize=(2,2)); a.imshow(test_maze.render_grid(), cmap='Greys', origin='upper'); a.set_title("Test Maze"); a.set_xticks([]); a.set_yticks([])
    cols[-1].pyplot(fig)

st.markdown("---")
st.caption("Notes: This demo uses a simplified meta-init by averaging short-trained Q-tables across tasks. It is designed to illustrate fast adaptation behavior visually (few-shot fine-tuning) and is ideal for demos and portfolios.")
