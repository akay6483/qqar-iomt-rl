import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import os

from algorithms.network import NetworkEnv
from experiment.simulation_engine import SimulationEngine
from algorithms.two_hop_qqar import QLearningAgent
from algorithms.one_hop_qqar import OneHopQQARAgent
from algorithms.plain_q_learning import PlainQLearningAgent
from algorithms.paper_config import (
    AGENT_LABELS,
    LABEL_BASELINE_Q,
    LABEL_QQAR_1HOP,
    LABEL_QQAR_2HOP,
    MAX_EPISODES,
)

def run_3way_topology_benchmark():
    NUM_RUNS = 10
    node_counts = [200, 400, 600, 800, 1000]
    agents_list = list(AGENT_LABELS)
    base_seed = 42
    
    final_results = {agent: {metric: [] for metric in ['pdr', 'delay', 'hops', 'energy']} for agent in agents_list}
    final_std = {agent: {metric: [] for metric in ['pdr', 'delay', 'hops', 'energy']} for agent in agents_list}
    
    for num_nodes in node_counts:
        print(f"\n{'='*50}")
        print(f" Routing algorithm comparison topology: {num_nodes} WBAN nodes")
        print(f"{'='*50}")
        
        runs_data = {agent: {'pdr': [], 'delay': [], 'hops': [], 'energy': []} for agent in agents_list}
        
        for run in range(NUM_RUNS):
            run_seed = base_seed + run
            
            random.seed(run_seed)
            base_env = NetworkEnv(area_size=500, num_nodes=num_nodes, tx_range=50)
            base_env.deploy_nodes()
            sinks = random.sample(range(num_nodes), max(10, int(num_nodes * 0.05)))
            wban_nodes = [n for n in range(num_nodes) if n not in sinks]

            agent_factories = {
                LABEL_QQAR_2HOP: QLearningAgent,
                LABEL_QQAR_1HOP: OneHopQQARAgent,
                LABEL_BASELINE_Q: PlainQLearningAgent
            }
            
            for agent_index, (agent_name, agent_cls) in enumerate(agent_factories.items()):
                env = base_env.clone_topology()
                for node_id, node in env.nodes.items():
                    node.broadcast_hello(1.0)
                discovery_energy = env.control_energy_consumed

                random.seed((run_seed * 1000) + agent_index)
                agent = agent_cls(env, max_episodes=MAX_EPISODES)
                agent.train(sinks)
                engine = SimulationEngine(
                    env,
                    agent,
                    sinks,
                    wban_nodes,
                    data_rate=5,
                    max_time=15.0,
                    traffic_seed=(run_seed * 100000) + 7,
                    channel_seed=(run_seed * 100000) + 13,
                    initial_energy=None,
                )
                pdr, delay, hops, energy = engine.run()
                
                runs_data[agent_name]['pdr'].append(pdr)
                runs_data[agent_name]['delay'].append(delay)
                runs_data[agent_name]['hops'].append(hops)
                runs_data[agent_name]['energy'].append(discovery_energy + agent.training_energy_consumed + energy)

        # Average Results
        for agent in agents_list:
            for metric in ['pdr', 'delay', 'hops', 'energy']:
                final_results[agent][metric].append(np.mean(runs_data[agent][metric]))
                final_std[agent][metric].append(np.std(runs_data[agent][metric]))

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Routing Algorithm Topology Evaluation', fontsize=16, fontweight='bold')

    colors = {LABEL_QQAR_2HOP: 'navy', LABEL_QQAR_1HOP: 'forestgreen', LABEL_BASELINE_Q: 'darkorange'}
    markers = {LABEL_QQAR_2HOP: 'd', LABEL_QQAR_1HOP: '^', LABEL_BASELINE_Q: 's'}

    # PDR
    for agent in agents_list:
        axs[0, 0].errorbar(node_counts, final_results[agent]['pdr'], yerr=final_std[agent]['pdr'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2, capsize=3)
    axs[0, 0].set(title='(a) PDR vs WBANs', xlabel='Number of WBANs', ylabel='PDR (%)')
    axs[0, 0].legend()

    # Delay
    for agent in agents_list:
        axs[0, 1].errorbar(node_counts, final_results[agent]['delay'], yerr=final_std[agent]['delay'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2, capsize=3)
    axs[0, 1].set(title='(b) Average E2E Delay vs WBANs', xlabel='Number of WBANs', ylabel='Delay (ms)')
    axs[0, 1].legend()

    # Hop Count
    x = np.arange(len(node_counts))
    width = 0.25
    axs[1, 0].bar(x - width, final_results[LABEL_QQAR_2HOP]['hops'], width, yerr=final_std[LABEL_QQAR_2HOP]['hops'], capsize=3, label=LABEL_QQAR_2HOP, color=colors[LABEL_QQAR_2HOP], edgecolor='black')
    axs[1, 0].bar(x, final_results[LABEL_QQAR_1HOP]['hops'], width, yerr=final_std[LABEL_QQAR_1HOP]['hops'], capsize=3, label=LABEL_QQAR_1HOP, color=colors[LABEL_QQAR_1HOP], edgecolor='black')
    axs[1, 0].bar(x + width, final_results[LABEL_BASELINE_Q]['hops'], width, yerr=final_std[LABEL_BASELINE_Q]['hops'], capsize=3, label=LABEL_BASELINE_Q, color=colors[LABEL_BASELINE_Q], edgecolor='black')
    axs[1, 0].set(title='(c) Average Hop Count', xlabel='Number of WBANs', ylabel='Hop Count')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(node_counts)
    axs[1, 0].legend()

    # Energy
    axs[1, 1].bar(x - width, final_results[LABEL_QQAR_2HOP]['energy'], width, yerr=final_std[LABEL_QQAR_2HOP]['energy'], capsize=3, label=LABEL_QQAR_2HOP, color=colors[LABEL_QQAR_2HOP], edgecolor='black')
    axs[1, 1].bar(x, final_results[LABEL_QQAR_1HOP]['energy'], width, yerr=final_std[LABEL_QQAR_1HOP]['energy'], capsize=3, label=LABEL_QQAR_1HOP, color=colors[LABEL_QQAR_1HOP], edgecolor='black')
    axs[1, 1].bar(x + width, final_results[LABEL_BASELINE_Q]['energy'], width, yerr=final_std[LABEL_BASELINE_Q]['energy'], capsize=3, label=LABEL_BASELINE_Q, color=colors[LABEL_BASELINE_Q], edgecolor='black')
    axs[1, 1].set(title='(d) Discovery + Training + Runtime Energy', xlabel='Number of WBANs', ylabel='Energy (J)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(node_counts)
    axs[1, 1].legend()

    for ax in axs.flat: ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'evaluation_topology_graphs.png')
    plt.savefig(save_path)
    print(f"\nSaved results to: {save_path}")

if __name__ == "__main__":
    run_3way_topology_benchmark()
