import random  # Added missing import
from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent

def setup_and_train():
    print("\n--- 1. Network Initialization ---")
    env = NetworkEnv(area_size=500, num_nodes=200, tx_range=50)
    env.deploy_nodes()
    
    # Notice: The artificial pinning of Node 0 and Node 199 has been removed!

    print("\n--- 2. Neighbor Discovery (Hello Packets) ---")
    for node_id, node in env.nodes.items():
        node.broadcast_hello(1.0)
    print("Discovery complete.")

    print("\n--- 3. Reinforcement Learning Training ---")

    sinks = random.sample(range(200), 10) # Randomly assign 10 nodes to act as sinks
    print(f"Designated Sink Nodes: {sinks}")
    
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, max_episodes=5000)
    agent.train(sinks)
    
    # Return the sinks array so the menu can use it for distance calculations
    return env, agent, sinks

def display_q_table(env, target_node_id, sinks):
    if target_node_id not in env.nodes:
        print("Error: Node ID does not exist.")
        return
        
    node = env.nodes[target_node_id]
    
    if target_node_id in sinks:
        print(f"\nNode {target_node_id} is a Sink Node. It does not route data further.")
        return
    
    if not hasattr(node, 'q_table') or not node.q_table:
        print(f"\nNode {target_node_id} has an empty Q-Table. (It may be isolated).")
        return
        
    print(f"\n==========================================")
    print(f" ROUTING TABLE FOR NODE {target_node_id}")
    print(f"==========================================")
    
    # Calculate distance to the *closest* sink for context
    min_dist = min([env.get_distance(target_node_id, s) for s in sinks])
    print(f"Distance to closest Sink: {min_dist:.2f}m")
    
    print(f"{'Next Hop ID':<15} | {'Q-Value':<10}")
    print("-" * 30)
    
    # Sort table so the best route is at the top
    sorted_q_table = sorted(node.q_table.items(), key=lambda item: item[1], reverse=True)
    
    for next_hop, q_val in sorted_q_table:
        marker = " <--- Best Route" if next_hop == sorted_q_table[0][0] else ""
        print(f"{next_hop:<15} | {q_val:<10.2f}{marker}")
    print("==========================================\n")

def menu():
    env, agent, sinks = setup_and_train()
    
    while True:
        print("Routing Table Explorer")
        print("----------------------")
        print("1. View Node 0 Table (Example Node)")
        print("2. View Specific Node Table")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            display_q_table(env, 0, sinks)
        elif choice == '2':
            try:
                node_id = int(input("Enter Node ID (0-199): "))
                display_q_table(env, node_id, sinks)
            except ValueError:
                print("Please enter a valid integer.")
        elif choice == '3':
            print("Exiting explorer.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    menu()
