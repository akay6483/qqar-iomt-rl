from algorithms.network import NetworkEnv

def main():
    # 1. Initialize and build the physical environment (200 nodes, 50m range)
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()
    
    print("\n--- Starting Phase 1: Network Discovery (Hello Packets) ---")
    current_time = 1.0 # Mocking a time slot
    
    # Every node broadcasts its initial Hello packet to the network
    for node_id, node in env.nodes.items():
        node.broadcast_hello(current_time)
        
    print("Network discovery complete. All routing tables updated.\n")
    
    # 2. Verify the results for a random node (e.g., Node 0)
    target_node = env.nodes[0]
    one_hop_count = 0
    two_hop_count = 0
    
    print(f"--- Routing Table for Node {target_node.node_id} ---")
    for neighbor_id, data in target_node.neighbor_list.items():
        if data['hops'] == 1:
            one_hop_count += 1
            print(f"Node {neighbor_id} is a 1-hop neighbor. (Energy: {data['energy']}J)")
        elif data['hops'] == 2:
            two_hop_count += 1
            print(f"Node {neighbor_id} is a 2-hop neighbor. (Energy: {data['energy']}J)")
            
    print(f"\nSummary for Node {target_node.node_id}:")
    print(f"Total 1-hop neighbors: {one_hop_count}")
    print(f"Total 2-hop neighbors: {two_hop_count}")

if __name__ == "__main__":
    main()
