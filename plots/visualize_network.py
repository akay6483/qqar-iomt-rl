import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import networkx as nx
from algorithms.network import NetworkEnv
import networkx as nx
from algorithms.network import NetworkEnv

def visualize_topology():
    print("Initializing network for visualization...")
    # Initialize the same environment you used in main.py
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()

    # Extract the (x,y) positions we assigned during deployment
    positions = nx.get_node_attributes(env.graph, 'pos')

    print("Generating plot...")
    # Set up the matplotlib figure size
    plt.figure(figsize=(10, 10))
    
    # 1. Draw the edges (communication links) with low opacity (alpha) so it's not too cluttered
    nx.draw_networkx_edges(env.graph, positions, alpha=0.15, edge_color='gray')
    
    # 2. Draw the WBAN nodes
    nx.draw_networkx_nodes(env.graph, positions, node_size=30, node_color='dodgerblue', edgecolors='black')
    
    # Optional: If you want to see the node IDs, uncomment the line below
    # nx.draw_networkx_labels(env.graph, positions, font_size=6, font_color='black')

    # Format the grid
    plt.title(f"IoMT WBAN Topology\n{env.num_nodes} Nodes, 50m Transmission Range", fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    
    # Force the axes to reflect the 500x500m physical room
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    
    # Turn on the grid for easier distance estimation
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    print("Displaying plot. Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    visualize_topology()
