import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np

from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent
from experiments.simulation_engine import SimulationEngine

def run_node_density_benchmark():
    node_counts = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    results_pdr, results_delay, results_ro, results_energy = [], [], [], []
    packets_to_simulate = 500
    
    for num_nodes in node_counts:
        print(f"Benchmarking Node Density: {num_nodes} WBANs...")
        env = NetworkEnv(area_size=500, num_nodes=num_nodes, tx_range=50)
        env.deploy_nodes()
        
        num_sinks = max(10, int(num_nodes * 0.05))
        sinks = random.sample(range(num_nodes), num_sinks)
        wban_nodes = [n for n in range(num_nodes) if n not in sinks]
        
        # Calculate Routing Overhead (RO) for Discovery
        # Hello packets (bytes) broadcasted by all nodes
        ro_bytes = num_nodes * 40 # Assume 40 bytes per Hello packet
        
        for node_id, node in env.nodes.items():
            node.broadcast_hello(1.0)
            
        agent = QLearningAgent(env, alpha=0.5, gamma=0.9, max_episodes=2000)
        agent.train(sinks)
        
        engine = SimulationEngine(env, agent, sinks, wban_nodes, data_rate=5, max_time=20.0)
        pdr, avg_delay, avg_hops, energy = engine.run()
        
        ro_bytes = num_nodes * 40 + (agent.max_episodes * 15)
        
        results_pdr.append(pdr)
        results_delay.append(avg_delay)
        results_ro.append(ro_bytes)
        results_energy.append(energy)
                

    # Plotting (Fig 7a, 8a, 9a, 11 equivalents)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance vs. Number of WBAN Nodes', fontsize=16, fontweight='bold')

    axs[0, 0].plot(node_counts, results_pdr, marker='d', color='navy')
    axs[0, 0].set(title='(a) PDR vs WBANs', xlabel='Number of WBANs', ylabel='PDR (%)')

    axs[0, 1].plot(node_counts, results_delay, marker='d', color='navy')
    axs[0, 1].set(title='(b) Average E2E Delay vs WBANs', xlabel='Number of WBANs', ylabel='Delay (ms)')

    axs[1, 0].plot(node_counts, results_ro, marker='d', color='navy')
    axs[1, 0].set(title='(c) Routing Overhead vs WBANs', xlabel='Number of WBANs', ylabel='RO (Bytes)')

    width = 40
    axs[1, 1].bar(np.array(node_counts), results_energy, width=width, color='dodgerblue', edgecolor='black')
    axs[1, 1].set(title='(d) Total Energy Consumption', xlabel='Number of WBANs', ylabel='Energy (J)')

    for ax in axs.flat: ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_node_density_benchmark()
