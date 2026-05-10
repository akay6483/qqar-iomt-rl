import matplotlib
matplotlib.use('Agg') # Headless to prevent GUI crashes
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent

def visualize_q_learning_path():
    print("1. Initializing network...")
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()

    sinks = random.sample(range(200), 10)
    wban_nodes = [n for n in range(200) if n not in sinks]

    print("2. Running Discovery...")
    for node_id, node in env.nodes.items():
        node.broadcast_hello(1.0)

    print(f"3. Training Q-Learning Agent with Sinks: {sinks}...")
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, max_episodes=5000)
    agent.train(sinks)

    print("4. Tracing optimal path from a random WBAN node...")
    start_id = random.choice(wban_nodes)
    path_nodes = [start_id]
    current_id = start_id
    visited = set([start_id])

    while current_id not in sinks:
        node = env.nodes[current_id]
        if not hasattr(node, 'q_table') or not node.q_table:
            print(f"ERROR: Path broke at node {current_id} (No Q-table).")
            break

        current_dist = agent.get_distance_to_closest_sink(current_id, sinks)
        candidates = [n for n in node.neighbor_list.keys() 
                      if current_dist > agent.get_distance_to_closest_sink(n, sinks)]

        if not candidates:
            print(f"ERROR: Path trapped at local minimum at node {current_id}.")
            break

        next_hop = agent.select_best_route(current_id, candidates, sinks)
        if next_hop in visited:
            print(f"ERROR: Routing loop detected at node {next_hop}!")
            break

        path_nodes.append(next_hop)
        visited.add(next_hop)
        current_id = next_hop

    print(f"\nSUCCESS: Data Packet Path: {path_nodes}")
    print(f"Total Hops: {len(path_nodes) - 1}")

    # --- Plotting ---
    positions = nx.get_node_attributes(env.graph, 'pos')
    plt.figure(figsize=(10, 10))

    nx.draw_networkx_edges(env.graph, positions, alpha=0.1, edge_color='gray')
    nx.draw_networkx_nodes(env.graph, positions, node_size=20, node_color='lightgray')

    # FIX: Use node_shape instead of marker for NetworkX
    nx.draw_networkx_nodes(env.graph, positions, nodelist=sinks, node_size=80, node_color='blue', node_shape='s', label='Sinks')

    if len(path_nodes) > 1:
        path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
        nx.draw_networkx_edges(env.graph, positions, edgelist=path_edges, edge_color='red', width=2.0)
        nx.draw_networkx_nodes(env.graph, positions, nodelist=[start_id], node_size=150, node_color='limegreen', node_shape='^', label=f'Source (Node {start_id})')

    plt.title("Multi-Sink Q-Learning Routing Path Verification", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'path_verification.png'))
    print("Plot saved to results/figures/path_verification.png")

if __name__ == "__main__":
    visualize_q_learning_path()
