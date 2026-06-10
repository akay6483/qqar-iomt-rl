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
)



def run_energy_by_episode_benchmark():
    num_runs = 10
    base_seed = 42
    num_nodes = 200
    area_size = 500
    tx_range = 50
    num_sinks = 10

    agent_factories = {
        LABEL_QQAR_2HOP: QLearningAgent,
        LABEL_QQAR_1HOP: OneHopQQARAgent,
        LABEL_BASELINE_Q: PlainQLearningAgent,
    }

    energy_runs = {agent: [] for agent in agent_factories}

    print(f"\n{'=' * 55}")
    print(" Routing algorithm comparison: cumulative training energy vs episodes")
    print(f"{'=' * 55}")

    for run in range(num_runs):
        run_seed = base_seed + run
        random.seed(run_seed)

        base_env = NetworkEnv(area_size=area_size, num_nodes=num_nodes, tx_range=tx_range)
        base_env.deploy_nodes()
        sinks = random.sample(range(num_nodes), num_sinks)

        for agent_index, (agent_name, agent_cls) in enumerate(agent_factories.items()):
            env = base_env.clone_topology()
            for node in env.nodes.values():
                node.broadcast_hello(1.0)

            random.seed((run_seed * 1000) + agent_index)
            agent = agent_cls(env, max_episodes=MAX_EPISODES)
            agent.train(sinks)

            if len(agent.training_energy_history) != MAX_EPISODES:
                raise RuntimeError(
                    f"{agent_name} produced {len(agent.training_energy_history)} energy samples, "
                    f"expected {MAX_EPISODES}."
                )
            energy_runs[agent_name].append(agent.training_energy_history)

    episodes = np.arange(1, MAX_EPISODES + 1)
    colors = {LABEL_QQAR_2HOP: 'navy', LABEL_QQAR_1HOP: 'forestgreen', LABEL_BASELINE_Q: 'darkorange'}

    plt.figure(figsize=(10, 6))
    for agent_name, runs in energy_runs.items():
        values = np.array(runs)
        mean_energy = values.mean(axis=0)
        std_energy = values.std(axis=0)

        plt.plot(episodes, mean_energy, color=colors[agent_name], linewidth=2, label=agent_name)
        plt.fill_between(
            episodes,
            mean_energy - std_energy,
            mean_energy + std_energy,
            color=colors[agent_name],
            alpha=0.15,
            linewidth=0,
        )

    plt.title('Cumulative Energy Consumption vs Training Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes')
    plt.ylabel('Cumulative Energy Consumption (J)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'energy_consumption_vs_episodes.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    run_energy_by_episode_benchmark()
