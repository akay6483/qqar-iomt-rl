import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
from algorithms.network import NetworkEnv
from algorithms.two_hop_qqar import QLearningAgent
from algorithms.paper_config import MAX_EPISODES, MAX_TRAINING_STEPS

def plot_convergence():
    random.seed(42)
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()
    
    for node_id, node in env.nodes.items():
        node.broadcast_hello(1.0)

    sinks = random.sample(range(200), 10)
    agent = QLearningAgent(env, max_episodes=MAX_EPISODES)
    wban_nodes = [n for n in range(200) if n not in sinks]
    
    cumulative_rewards = []
    
    # Custom training loop to track rewards per episode
    for episode in range(agent.max_episodes):
        current_id = random.choice(wban_nodes)
        pkt = agent._sample_training_packet(episode + 1, current_id, sinks)
        steps = 0
        episode_reward = 0
        epsilon = max(0.1, 1.0 - episode / (agent.max_episodes * 0.5))
        
        while current_id not in sinks and steps < MAX_TRAINING_STEPS:
            current_node = env.nodes[current_id]
            current_dist = min([env.get_distance(current_id, s) for s in sinks])
            candidates = [n for n in current_node.neighbor_list.keys() 
                          if current_dist > min([env.get_distance(n, s) for s in sinks])]
                          
            if not candidates:
                episode_reward -= 100 # Penalty for dead end
                break 
                
            if random.random() < epsilon:
                action_id = random.choice(candidates)
            else:
                action_id = max(candidates, key=lambda a: agent.get_q_value(current_id, a))
                
            success = agent._apply_training_route(current_id, action_id)
            reward = agent.calculate_reward(current_id, action_id, sinks, candidates, pkt)
            if not success:
                reward = -100.0
            episode_reward += reward
            
            if action_id in sinks:
                max_next_q = 0.0
            else:
                next_node = env.nodes[action_id]
                next_dist = min([env.get_distance(action_id, s) for s in sinks])
                next_candidates = [n for n in next_node.neighbor_list.keys() 
                                   if next_dist > min([env.get_distance(n, s) for s in sinks])]
                max_next_q = max([agent.get_q_value(action_id, a) for a in next_candidates]) if next_candidates else 0.0
                
            old_q = agent.get_q_value(current_id, action_id)
            new_q = old_q + agent.alpha * (reward + agent.gamma * max_next_q - old_q)
            agent.set_q_value(current_id, action_id, new_q)
            agent._record_q_update_overhead(current_id, action_id)
            
            if not success:
                break
            current_id = action_id
            steps += 1
            
        cumulative_rewards.append(episode_reward)

    # Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(agent.max_episodes), cumulative_rewards, color='navy', label='QQAR')
    plt.title("Algorithm Convergence (Cumulative Reward vs Episodes)", fontweight='bold')
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'q_learning_convergence.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_convergence()
