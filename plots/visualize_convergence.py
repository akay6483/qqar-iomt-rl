import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif', 'serif'],
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.autolayout': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black'
})
import numpy as np
import os
import random

from algorithms.network import NetworkEnv
from algorithms.one_hop_qqar import OneHopQQARAgent
from algorithms.plain_q_learning import PlainQLearningAgent
from algorithms.two_hop_qqar import QLearningAgent

from algorithms.paper_config import (
    LABEL_BASELINE_Q,
    LABEL_QQAR_1HOP,
    LABEL_QQAR_2HOP,
    MAX_EPISODES,
    MAX_TRAINING_STEPS,
)

def evaluate_greedy_policy(agent, env, sinks, wban_nodes, agent_label):
    """
    Evaluates the Q-table with NO training wheels (distance filters removed).
    Tests all nodes to generate a perfectly smooth percentage curve.
    """
    total_eval_reward = 0.0
    
    for start_node in wban_nodes:
        current_id = start_node
        steps = 0
        success = False
        visited = set() # Prevent infinite looping for untrained agents
        
        while current_id not in sinks and steps < MAX_TRAINING_STEPS:
            visited.add(current_id)
            current_node = env.nodes[current_id]
            
            # REMOVED the distance filter. If it hasn't learned to go forward, it fails.
            if agent_label == LABEL_QQAR_2HOP:
                candidates = [n for n in current_node.neighbor_list.keys() if n not in visited]
            else:
                candidates = [n for n, data in current_node.neighbor_list.items() if data['hops'] == 1 and n not in visited]
                
            if not candidates:
                break # Wandered into a dead end, fails.
                
            # Pick highest Q-value. If empty (Episode 0), it picks randomly and likely fails.
            action_id = max(candidates, key=lambda a: agent.get_q_value(current_id, a))
            
            if action_id in sinks:
                success = True
                break
                
            current_id = action_id
            steps += 1
            
        if success:
            total_eval_reward += 100.0
        
    return total_eval_reward / len(wban_nodes)


def run_convergence_benchmark():
    num_runs = 5 # Set to 5 for speed, 10 for tighter error bands
    base_seed = 42
    num_nodes = 200
    
    agent_factories = {
        LABEL_QQAR_2HOP: QLearningAgent,
        LABEL_QQAR_1HOP: OneHopQQARAgent,
        LABEL_BASELINE_Q: PlainQLearningAgent,
    }

    reward_runs = {agent: [] for agent in agent_factories}

    print(f"\n{'=' * 55}")
    print(" Routing algorithm comparison: True RL Convergence")
    print(f"{'=' * 55}")

    for run in range(num_runs):
        run_seed = base_seed + run
        random.seed(run_seed)

        base_env = NetworkEnv(area_size=500, num_nodes=num_nodes, tx_range=50)
        base_env.deploy_nodes()
        sinks = random.sample(range(num_nodes), 10)
        wban_nodes = [n for n in range(num_nodes) if n not in sinks]

        for agent_index, (agent_name, agent_cls) in enumerate(agent_factories.items()):
            env = base_env.clone_topology()
            for node in env.nodes.values():
                node.broadcast_hello(1.0)

            random.seed((run_seed * 1000) + agent_index)
            agent = agent_cls(env, max_episodes=MAX_EPISODES)
            cumulative_rewards = []
            
            for episode in range(agent.max_episodes):
                # FIX 3: Reset Energy and Congestion for a fresh episodic start
                for node in env.nodes.values():
                    node.energy = 100.0
                    node.avg_traffic_load = 0.0

                # 1. Training Step (With Distance Filters and Epsilon Randomness)
                current_id = random.choice(wban_nodes)
                pkt = agent._sample_training_packet(episode + 1, current_id, sinks)
                steps = 0
                epsilon = max(0.1, 1.0 - episode / (agent.max_episodes * 0.5))
                
                while current_id not in sinks and steps < MAX_TRAINING_STEPS:
                    current_node = env.nodes[current_id]
                    current_dist = min([env.get_distance(current_id, s) for s in sinks])
                    
                    if agent_name == LABEL_QQAR_2HOP:
                        candidates = [n for n in current_node.neighbor_list.keys() if current_dist > min([env.get_distance(n, s) for s in sinks])]
                    else:
                        candidates = [n for n, data in current_node.neighbor_list.items() if data['hops'] == 1 and current_dist > min([env.get_distance(n, s) for s in sinks])]
                    
                    if not candidates: break 
                    
                    if random.random() < epsilon: action_id = random.choice(candidates)
                    else: action_id = max(candidates, key=lambda a: agent.get_q_value(current_id, a))
                        
                    if agent_name == LABEL_BASELINE_Q:
                        reward = 100.0 if action_id in sinks else 0.0
                    else:
                        reward = agent.calculate_reward(current_id, action_id, sinks, candidates, pkt)
                        
                    if action_id in sinks: max_next_q = 0.0
                    else:
                        next_node = env.nodes[action_id]
                        next_dist = min([env.get_distance(action_id, s) for s in sinks])
                        if agent_name == LABEL_QQAR_2HOP: next_cands = [n for n in next_node.neighbor_list.keys() if next_dist > min([env.get_distance(n, s) for s in sinks])]
                        else: next_cands = [n for n, d in next_node.neighbor_list.items() if d['hops'] == 1 and next_dist > min([env.get_distance(n, s) for s in sinks])]
                        max_next_q = max([agent.get_q_value(action_id, a) for a in next_cands]) if next_cands else 0.0
                        
                    old_q = agent.get_q_value(current_id, action_id)
                    new_q = old_q + agent.alpha * (reward + agent.gamma * max_next_q - old_q)
                    agent.set_q_value(current_id, action_id, new_q)
                    current_id = action_id
                    steps += 1
                    
                # 2. Evaluation Step (Every 10 episodes to save calculation time)
                if episode % 10 == 0:
                    eval_score = evaluate_greedy_policy(agent, env, sinks, wban_nodes, agent_name)
                    cumulative_rewards.append(eval_score)
                    
            full_rewards = np.interp(range(MAX_EPISODES), range(0, MAX_EPISODES, 10), cumulative_rewards)
            reward_runs[agent_name].append(full_rewards)

    # Plotting
    episodes = np.arange(1, MAX_EPISODES + 1)
    colors = {LABEL_QQAR_2HOP: 'navy', LABEL_QQAR_1HOP: 'forestgreen', LABEL_BASELINE_Q: 'darkorange'}

    plt.figure(figsize=(10, 6))
    for agent_name, runs in reward_runs.items():
        values = np.array(runs)
        mean_reward = values.mean(axis=0)
        std_reward = values.std(axis=0)

        plt.plot(episodes, mean_reward, color=colors[agent_name], linewidth=2, label=agent_name)
        plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, color=colors[agent_name], alpha=0.15, linewidth=0)

    plt.title('Algorithm Convergence (Greedy Policy vs Episodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes')
    plt.ylabel('Delivery Success Score (%)')
    
    # Anchored Y-Axis to match the paper
    plt.ylim(0, 105)
    plt.xlim(0, MAX_EPISODES)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'q_learning_convergence.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved results to: {save_path}")

if __name__ == "__main__":
    run_convergence_benchmark()
