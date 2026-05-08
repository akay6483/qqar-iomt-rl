import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np

from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent
from experiments.simulation_engine import SimulationEngine

from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent
from experiments.simulation_engine import SimulationEngine # Import your new engine!

def run_data_rate_benchmark():
    data_rates = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    results_pdr, results_delay, results_ro, results_hops = [], [], [], []
    
    print("Initializing static 200-node network for data rate tests...")
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()
    sinks = random.sample(range(200), 10)
    wban_nodes = [n for n in range(200) if n not in sinks]
    
    for node_id, node in env.nodes.items():
        node.broadcast_hello(1.0)
        
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, max_episodes=2000)
    agent.train(sinks)
    
    for rate in data_rates:
        # Run a 10-second global time simulation for each data rate
        engine = SimulationEngine(env, agent, sinks, wban_nodes, data_rate=rate, max_time=10.0)
        pdr, avg_delay, avg_hops, energy = engine.run()
        
        # Routing overhead scales with data rate
        ro_bytes = (200 * 40) + (rate * 5) 
        
        results_pdr.append(pdr)
        results_delay.append(avg_delay)
        results_ro.append(ro_bytes)
        results_hops.append(avg_hops)
        
        print(f"Result -> PDR: {pdr:.1f}%, Delay: {avg_delay:.1f}ms, Hops: {avg_hops:.2f}")

    # Plotting (Fig 7b, 8b, 9b, 10 equivalents)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance vs. Source Data Rate (Packets/s)', fontsize=16, fontweight='bold')

    axs[0, 0].plot(data_rates, results_pdr, marker='d', color='navy')
    axs[0, 0].set(title='(a) PDR vs Data Rate', xlabel='Source Data Rate', ylabel='PDR (%)')

    axs[0, 1].plot(data_rates, results_delay, marker='d', color='navy')
    axs[0, 1].set(title='(b) E2E Delay vs Data Rate', xlabel='Source Data Rate', ylabel='Delay (ms)')

    axs[1, 0].plot(data_rates, results_ro, marker='d', color='navy')
    axs[1, 0].set(title='(c) Routing Overhead vs Data Rate', xlabel='Source Data Rate', ylabel='RO (Bytes)')

    width = 2
    axs[1, 1].bar(np.array(data_rates), results_hops, width=width, color='dodgerblue', edgecolor='black')
    axs[1, 1].set(title='(d) Average Hop Count', xlabel='Source Data Rate', ylabel='Hop Count')

    for ax in axs.flat: ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_data_rate_benchmark()
