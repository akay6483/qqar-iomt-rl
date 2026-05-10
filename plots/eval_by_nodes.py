import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import os

from algorithms.network import NetworkEnv
from experiments.simulation_engine import SimulationEngine
from algorithms.q_learning import QLearningAgent 
from algorithms.one_hop_qqar import OneHopQQARAgent 
from algorithms.plain_q_learning import PlainQLearningAgent 

def run_3way_reproducible_benchmark():
    # OPTIMIZATIONS FOR SPEED
    NUM_RUNS = 1 
    node_counts = [200, 400, 600, 800, 1000]
    agents_list = ['QQAR (2-Hop)', 'QQAR (1-Hop)', 'Plain Q (1-Hop)']
    
    final_results = {agent: {'pdr': [], 'delay': [], 'ro': [], 'energy': []} for agent in agents_list}
    
    for num_nodes in node_counts:
        print(f"\n{'='*50}")
        print(f" FAST BENCHMARK: {num_nodes} WBAN Nodes")
        print(f"{'='*50}")
        
        runs_data = {agent: {'pdr': [], 'delay': [], 'ro': [], 'energy': []} for agent in agents_list}
        
        for run in range(NUM_RUNS):
            random.seed(42 + run)
            
            env = NetworkEnv(area_size=500, num_nodes=num_nodes, tx_range=50)
            env.deploy_nodes()
            num_sinks = max(10, int(num_nodes * 0.05))
            sinks = random.sample(range(num_nodes), num_sinks)
            wban_nodes = [n for n in range(num_nodes) if n not in sinks]
            
            for node_id, node in env.nodes.items():
                node.broadcast_hello(1.0)
            ro_base = num_nodes * 40
            
            agent_instances = {
                'QQAR (2-Hop)': QLearningAgent(env, max_episodes=1000), # Cut training time in half
                'QQAR (1-Hop)': OneHopQQARAgent(env, max_episodes=1000),
                'Plain Q (1-Hop)': PlainQLearningAgent(env, max_episodes=1000)
            }
            
            for agent_name, agent in agent_instances.items():
                agent.train(sinks)
                # Max time back to 15.0 seconds
                engine = SimulationEngine(env, agent, sinks, wban_nodes, data_rate=5, max_time=15.0)
                pdr, delay, hops, energy = engine.run()
                
                runs_data[agent_name]['pdr'].append(pdr)
                runs_data[agent_name]['delay'].append(delay)
                runs_data[agent_name]['ro'].append(ro_base + (agent.max_episodes * 15 if 'QQAR' in agent_name else 10))
                runs_data[agent_name]['energy'].append(energy)

        for agent_name in agents_list:
            final_results[agent_name]['pdr'].append(np.mean(runs_data[agent_name]['pdr']))
            final_results[agent_name]['delay'].append(np.mean(runs_data[agent_name]['delay']))
            final_results[agent_name]['ro'].append(np.mean(runs_data[agent_name]['ro']))
            final_results[agent_name]['energy'].append(np.mean(runs_data[agent_name]['energy']))

    # --- Plotting the 3-Way Comparison ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Protocol Performance Comparison', fontsize=16, fontweight='bold')

    colors = {'QQAR (2-Hop)': 'navy', 'QQAR (1-Hop)': 'forestgreen', 'Plain Q (1-Hop)': 'darkorange'}
    markers = {'QQAR (2-Hop)': 'd', 'QQAR (1-Hop)': '^', 'Plain Q (1-Hop)': 's'}

    # PDR
    for agent in agents_list:
        axs[0, 0].plot(node_counts, final_results[agent]['pdr'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[0, 0].set(title='(a) Packet Delivery Ratio vs WBANs', xlabel='Number of WBANs', ylabel='PDR (%)')
    axs[0, 0].legend()

    # Delay
    for agent in agents_list:
        axs[0, 1].plot(node_counts, final_results[agent]['delay'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[0, 1].set(title='(b) Average E2E Delay vs WBANs', xlabel='Number of WBANs', ylabel='Delay (ms)')
    axs[0, 1].legend()

    # Routing Overhead
    for agent in agents_list:
        axs[1, 0].plot(node_counts, final_results[agent]['ro'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[1, 0].set(title='(c) Routing Overhead vs WBANs', xlabel='Number of WBANs', ylabel='RO (Bytes)')
    axs[1, 0].legend()

    # Energy Consumption 
    x = np.arange(len(node_counts))
    width = 0.25
    axs[1, 1].bar(x - width, final_results['QQAR (2-Hop)']['energy'], width, label='QQAR (2-Hop)', color=colors['QQAR (2-Hop)'], edgecolor='black')
    axs[1, 1].bar(x, final_results['QQAR (1-Hop)']['energy'], width, label='QQAR (1-Hop)', color=colors['QQAR (1-Hop)'], edgecolor='black')
    axs[1, 1].bar(x + width, final_results['Plain Q (1-Hop)']['energy'], width, label='Plain Q (1-Hop)', color=colors['Plain Q (1-Hop)'], edgecolor='black')
    axs[1, 1].set(title='(d) Total Energy Consumption', xlabel='Number of WBANs', ylabel='Energy (J)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(node_counts)
    axs[1, 1].legend()

    for ax in axs.flat: 
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'algorithm_3way_comparison.png')
    plt.savefig(save_path)
    print(f"\n>>> Simulation fully complete! Results saved to: {save_path} <<<")

if __name__ == "__main__":
    run_3way_reproducible_benchmark()
