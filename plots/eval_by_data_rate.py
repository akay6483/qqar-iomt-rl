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

def run_3way_data_rate_benchmark():
    NUM_RUNS = 1 # OPTIMIZED FOR SPEED
    data_rates = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    agents_list = ['QQAR (2-Hop)', 'QQAR (1-Hop)', 'Plain Q (1-Hop)']
    
    final_results = {agent: {metric: [] for metric in ['pdr', 'delay', 'ro', 'hops']} for agent in agents_list}
    
    print(f"\n{'='*50}")
    print(f" FAST BENCHMARK: Data Rates (5-50 pkts/s)")
    print(f"{'='*50}")

    runs_data = {rate: {agent: {'pdr': [], 'delay': [], 'ro': [], 'hops': []} for agent in agents_list} for rate in data_rates}

    for run in range(NUM_RUNS):
        random.seed(42 + run)
        
        # 1. Build a single, consistent 200-node environment for this run
        env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
        env.deploy_nodes()
        sinks = random.sample(range(200), 10)
        wban_nodes = [n for n in range(200) if n not in sinks]
        
        for node_id, node in env.nodes.items():
            node.broadcast_hello(1.0)
            
        ro_discovery = 200 * 40
        
        # 2. Train agents ONCE per topology
        agent_instances = {
            'QQAR (2-Hop)': QLearningAgent(env, max_episodes=1000), # OPTIMIZED
            'QQAR (1-Hop)': OneHopQQARAgent(env, max_episodes=1000), # OPTIMIZED
            'Plain Q (1-Hop)': PlainQLearningAgent(env, max_episodes=1000) # OPTIMIZED
        }
        
        for agent_name, agent in agent_instances.items():
            agent.train(sinks)
            
        # 3. Test all data rates
        for rate in data_rates:
            for agent_name, agent in agent_instances.items():
                engine = SimulationEngine(env, agent, sinks, wban_nodes, data_rate=rate, max_time=15.0) # OPTIMIZED
                pdr, delay, hops, _ = engine.run()
                
                training_ro = agent.max_episodes * (15 if 'QQAR' in agent_name else 10)
                traffic_ro = rate * 5 
                
                runs_data[rate][agent_name]['pdr'].append(pdr)
                runs_data[rate][agent_name]['delay'].append(delay)
                runs_data[rate][agent_name]['ro'].append(ro_discovery + training_ro + traffic_ro)
                runs_data[rate][agent_name]['hops'].append(hops)

    # 4. Average Results 
    for rate in data_rates:
        for agent in agents_list:
            final_results[agent]['pdr'].append(np.mean(runs_data[rate][agent]['pdr']))
            final_results[agent]['delay'].append(np.mean(runs_data[rate][agent]['delay']))
            final_results[agent]['ro'].append(np.mean(runs_data[rate][agent]['ro']))
            final_results[agent]['hops'].append(np.mean(runs_data[rate][agent]['hops']))

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Protocol Performance vs. Data Rate (Fast Benchmark)', fontsize=16, fontweight='bold')

    colors = {'QQAR (2-Hop)': 'navy', 'QQAR (1-Hop)': 'forestgreen', 'Plain Q (1-Hop)': 'darkorange'}
    markers = {'QQAR (2-Hop)': 'd', 'QQAR (1-Hop)': '^', 'Plain Q (1-Hop)': 's'}

    for agent in agents_list:
        axs[0, 0].plot(data_rates, final_results[agent]['pdr'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[0, 0].set(title='(a) PDR vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='PDR (%)')
    axs[0, 0].legend()

    for agent in agents_list:
        axs[0, 1].plot(data_rates, final_results[agent]['delay'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[0, 1].set(title='(b) Average E2E Delay vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='Delay (ms)')
    axs[0, 1].legend()

    for agent in agents_list:
        axs[1, 0].plot(data_rates, final_results[agent]['ro'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2)
    axs[1, 0].set(title='(c) Routing Overhead vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='RO (Bytes)')
    axs[1, 0].legend()

    x = np.arange(len(data_rates))
    width = 0.25
    axs[1, 1].bar(x - width, final_results['QQAR (2-Hop)']['hops'], width, label='QQAR (2-Hop)', color=colors['QQAR (2-Hop)'], edgecolor='black')
    axs[1, 1].bar(x, final_results['QQAR (1-Hop)']['hops'], width, label='QQAR (1-Hop)', color=colors['QQAR (1-Hop)'], edgecolor='black')
    axs[1, 1].bar(x + width, final_results['Plain Q (1-Hop)']['hops'], width, label='Plain Q (1-Hop)', color=colors['Plain Q (1-Hop)'], edgecolor='black')
    axs[1, 1].set(title='(d) Average Hop Count', xlabel='Source Data Rate (pkts/s)', ylabel='Hop Count')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(data_rates)
    axs[1, 1].legend()

    for ax in axs.flat: ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data_rate_3way_comparison.png')
    plt.savefig(save_path)
    print(f"\n>>> Simulation complete! Results saved to: {save_path} <<<")

if __name__ == "__main__":
    run_3way_data_rate_benchmark()
