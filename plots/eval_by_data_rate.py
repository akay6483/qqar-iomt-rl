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

def run_3way_data_rate_benchmark():
    NUM_RUNS = 10
    data_rates = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    agents_list = list(AGENT_LABELS)
    base_seed = 42
    
    final_results = {agent: {metric: [] for metric in ['pdr', 'delay', 'ro', 'hops']} for agent in agents_list}
    final_std = {agent: {metric: [] for metric in ['pdr', 'delay', 'ro', 'hops']} for agent in agents_list}
    
    print(f"\n{'='*50}")
    print(f" Routing algorithm comparison: data rates 5-50 packets/s")
    print(f"{'='*50}")

    runs_data = {rate: {agent: {'pdr': [], 'delay': [], 'ro': [], 'hops': []} for agent in agents_list} for rate in data_rates}

    for run in range(NUM_RUNS):
        run_seed = base_seed + run
        random.seed(run_seed)
        
        # 1. Build one base topology per seed. Each agent receives a fresh clone.
        base_env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
        base_env.deploy_nodes()
        sinks = random.sample(range(200), 10)
        wban_nodes = [n for n in range(200) if n not in sinks]

        agent_factories = {
            LABEL_QQAR_2HOP: QLearningAgent,
            LABEL_QQAR_1HOP: OneHopQQARAgent,
            LABEL_BASELINE_Q: PlainQLearningAgent
        }

        trained_agents = {}
        overhead_by_agent = {}
        trained_energy_by_agent = {}

        # 2. Train each agent on the same topology, with isolated dynamic state.
        for agent_index, (agent_name, agent_cls) in enumerate(agent_factories.items()):
            env = base_env.clone_topology()

            for node_id, node in env.nodes.items():
                node.broadcast_hello(1.0)
            discovery_ro = env.routing_overhead_bytes

            random.seed((run_seed * 1000) + agent_index)
            agent = agent_cls(env, max_episodes=MAX_EPISODES)
            agent.train(sinks)

            trained_agents[agent_name] = agent
            overhead_by_agent[agent_name] = discovery_ro + agent.routing_overhead_bytes
            trained_energy_by_agent[agent_name] = {
                node_id: node.energy for node_id, node in agent.env.nodes.items()
            }
            
        # 3. Test all data rates
        for rate in data_rates:
            for agent_name, agent in trained_agents.items():
                traffic_seed = (run_seed * 100000) + (rate * 100) + 7
                channel_seed = (run_seed * 100000) + (rate * 100) + 13
                engine = SimulationEngine(
                    agent.env,
                    agent,
                    sinks,
                    wban_nodes,
                    data_rate=rate,
                    max_time=15.0,
                    traffic_seed=traffic_seed,
                    channel_seed=channel_seed,
                    initial_energy=trained_energy_by_agent[agent_name],
                )
                pdr, delay, hops, _ = engine.run()
                
                runs_data[rate][agent_name]['pdr'].append(pdr)
                runs_data[rate][agent_name]['delay'].append(delay)
                runs_data[rate][agent_name]['ro'].append(overhead_by_agent[agent_name] + engine.routing_overhead_bytes)
                runs_data[rate][agent_name]['hops'].append(hops)

    # 4. Average Results 
    for rate in data_rates:
        for agent in agents_list:
            for metric in ['pdr', 'delay', 'ro', 'hops']:
                final_results[agent][metric].append(np.mean(runs_data[rate][agent][metric]))
                final_std[agent][metric].append(np.std(runs_data[rate][agent][metric]))

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Routing Algorithm Performance vs. Data Rate', fontsize=16, fontweight='bold')

    colors = {LABEL_QQAR_2HOP: 'navy', LABEL_QQAR_1HOP: 'forestgreen', LABEL_BASELINE_Q: 'darkorange'}
    markers = {LABEL_QQAR_2HOP: 'd', LABEL_QQAR_1HOP: '^', LABEL_BASELINE_Q: 's'}

    for agent in agents_list:
        axs[0, 0].errorbar(data_rates, final_results[agent]['pdr'], yerr=final_std[agent]['pdr'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2, capsize=3)
    axs[0, 0].set(title='(a) PDR vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='PDR (%)')
    axs[0, 0].legend()

    for agent in agents_list:
        axs[0, 1].errorbar(data_rates, final_results[agent]['delay'], yerr=final_std[agent]['delay'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2, capsize=3)
    axs[0, 1].set(title='(b) Average E2E Delay vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='Delay (ms)')
    axs[0, 1].legend()

    for agent in agents_list:
        axs[1, 0].errorbar(data_rates, final_results[agent]['ro'], yerr=final_std[agent]['ro'], marker=markers[agent], color=colors[agent], label=agent, linewidth=2, capsize=3)
    axs[1, 0].set(title='(c) Routing Overhead vs Data Rate', xlabel='Source Data Rate (pkts/s)', ylabel='RO (Bytes)')
    axs[1, 0].legend()

    x = np.arange(len(data_rates))
    width = 0.25
    axs[1, 1].bar(x - width, final_results[LABEL_QQAR_2HOP]['hops'], width, yerr=final_std[LABEL_QQAR_2HOP]['hops'], capsize=3, label=LABEL_QQAR_2HOP, color=colors[LABEL_QQAR_2HOP], edgecolor='black')
    axs[1, 1].bar(x, final_results[LABEL_QQAR_1HOP]['hops'], width, yerr=final_std[LABEL_QQAR_1HOP]['hops'], capsize=3, label=LABEL_QQAR_1HOP, color=colors[LABEL_QQAR_1HOP], edgecolor='black')
    axs[1, 1].bar(x + width, final_results[LABEL_BASELINE_Q]['hops'], width, yerr=final_std[LABEL_BASELINE_Q]['hops'], capsize=3, label=LABEL_BASELINE_Q, color=colors[LABEL_BASELINE_Q], edgecolor='black')
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
    print(f"\nSaved results to: {save_path}")

if __name__ == "__main__":
    run_3way_data_rate_benchmark()
