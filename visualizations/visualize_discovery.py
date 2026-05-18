import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
from algorithms.network import NetworkEnv

def visualize_logical_topology(target_id=0):
    print("Initializing network and running discovery...")
    random.seed(42)
    # 1. Initialize and build physical environment
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()
    
    # 2. Run Phase 1: Network Discovery (Hello Packets)
    current_time = 1.0
    for node_id, node in env.nodes.items():
        node.broadcast_hello(current_time)
        
    print(f"Discovery complete. Visualizing logical topology for Node {target_id}...")
    target_node = env.nodes[target_id]
    
    # 3. Categorize nodes based on the target node's routing table
    one_hop_nodes = []
    two_hop_nodes = []
    other_nodes = []
    
    for n_id in env.graph.nodes():
        if n_id == target_id:
            continue
        # Check if the node is in the target's discovered neighbor list
        if n_id in target_node.neighbor_list:
            if target_node.neighbor_list[n_id]['hops'] == 1:
                one_hop_nodes.append(n_id)
            elif target_node.neighbor_list[n_id]['hops'] == 2:
                two_hop_nodes.append(n_id)
        else:
            other_nodes.append(n_id)
            
    # Extract positions
    positions = nx.get_node_attributes(env.graph, 'pos')
    
    plt.figure(figsize=(10, 10))
    
    # 4. Draw all underlying physical edges very lightly for context
    nx.draw_networkx_edges(env.graph, positions, alpha=0.1, edge_color='lightgray')
    
    # 5. Highlight the logical paths (Edges)
    # Draw direct paths from Source to 1-hop neighbors
    one_hop_edges = [(target_id, n) for n in one_hop_nodes]
    nx.draw_networkx_edges(env.graph, positions, edgelist=one_hop_edges, alpha=0.8, edge_color='limegreen', width=2.0)
    
    # Draw paths from 1-hop neighbors to 2-hop neighbors to show *how* they were discovered
    two_hop_edges = []
    for n2 in two_hop_nodes:
        relay_id = target_node.neighbor_list[n2].get('relay')
        if relay_id is not None and env.graph.has_edge(relay_id, n2):
            two_hop_edges.append((relay_id, n2))
                
    nx.draw_networkx_edges(env.graph, positions, edgelist=two_hop_edges, alpha=0.6, edge_color='orange', width=1.5)
    
    # 6. Draw Nodes with Color Coding
    # Undiscovered/Irrelevant nodes
    nx.draw_networkx_nodes(env.graph, positions, nodelist=other_nodes, node_size=20, node_color='lightgray', edgecolors='black', alpha=0.5, label='Other Nodes')
    # 2-Hop Neighbors
    nx.draw_networkx_nodes(env.graph, positions, nodelist=two_hop_nodes, node_size=40, node_color='orange', edgecolors='black', label='2-Hop Neighbors')
    # 1-Hop Neighbors
    nx.draw_networkx_nodes(env.graph, positions, nodelist=one_hop_nodes, node_size=60, node_color='limegreen', edgecolors='black', label='1-Hop Neighbors')
    # The Source Node
    nx.draw_networkx_nodes(env.graph, positions, nodelist=[target_id], node_size=120, node_color='red', edgecolors='black', label='Source Node')
    
    # Format the grid and legend
    plt.title(f"Logical Network View (Post-Discovery)\nPerspective of Source Node {target_id}", fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.legend(scatterpoints=1, loc='upper right')

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'discovery_node_{target_id}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # You can change the target_id to view the network from the perspective of any other node
    visualize_logical_topology(target_id=0)
