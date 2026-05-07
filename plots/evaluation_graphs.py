import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np

# Adjust imports to match your new directory structure
from algorithms.network import NetworkEnv
from algorithms.q_learning import QLearningAgent

def run_benchmark():
    node_counts = [200, 400, 600, 800, 1000]
    
    # Results arrays
    results_pdr = []
    results_delay = []
    results_hops = []
    results_energy = []
    
    packets_to_simulate = 500
    
    for num_nodes in node_counts:
        print(f"\n--- Benchmarking QQAR with {num_nodes} nodes ---")
        env = NetworkEnv(area_size=500, num_nodes=num_nodes, tx_range=50)
        env.deploy_nodes()
        
        # 5% sinks as per the paper
        num_sinks = max(10, int(num_nodes * 0.05))
        sinks = random.sample(range(num_nodes), num_sinks)
        wban_nodes = [n for n in range(num_nodes) if n not in sinks]
        
        for node_id, node in env.nodes.items():
            node.broadcast_hello(1.0)
            
        agent = QLearningAgent(env, alpha=0.5, gamma=0.9, max_episodes=2000)
        agent.train(sinks)
        
        # --- Run Packet Simulation ---
        successful_packets = 0
        total_hops = 0
        total_delay = 0.0
        energy_consumed = 0.0
        
        for _ in range(packets_to_simulate):
            current_id = random.choice(wban_nodes)
            hops = 0
            visited = set([current_id])
            packet_dropped = False
            
            while current_id not in sinks:
                node = env.nodes[current_id]
                if not hasattr(node, 'q_table') or not node.q_table:
                    packet_dropped = True
                    break
                    
                current_dist = agent.get_distance_to_closest_sink(current_id, sinks)
                candidates = [n for n in node.neighbor_list.keys() if current_dist > agent.get_distance_to_closest_sink(n, sinks)]
                
                if not candidates:
                    packet_dropped = True
                    break
                    
                next_hop = agent.select_best_route(current_id, candidates, sinks)
                
                if next_hop in visited:
                    packet_dropped = True # Routing loop
                    break
                    
                # Simulate energy drain (Transmission + Reception)
                env.nodes[current_id].record_transmission(success=True)  # Tx power (Paper Table 4)
                env.nodes[next_hop].energy -= 0.395   # Rx power (Paper Table 4)
                env.nodes[next_hop].pkt_in += 1;
                energy_consumed += 1.055
                
                # Accumulate delay (Placeholder 10ms per hop)
                total_delay += 10.0 
                
                visited.add(next_hop)
                current_id = next_hop
                hops += 1
                
                if hops > 50: # TTL expired
                    packet_dropped = True
                    break
                    
            if not packet_dropped:
                successful_packets += 1
                total_hops += hops
                
        # Calculate Metrics
        pdr = (successful_packets / packets_to_simulate) * 100
        avg_hops = total_hops / max(1, successful_packets)
        avg_delay = total_delay / max(1, successful_packets)
        
        results_pdr.append(pdr)
        results_delay.append(avg_delay)
        results_hops.append(avg_hops)
        results_energy.append(energy_consumed)
        
        print(f"PDR: {pdr:.1f}%, Avg Hops: {avg_hops:.2f}, Avg Delay: {avg_delay:.1f}ms, Energy: {energy_consumed:.1f}J")

    # --- Plotting the 4x4 Dashboard ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('QQAR Performance Evaluation (Isolated)', fontsize=16, fontweight='bold')

    # 1. PDR
    axs[0, 0].plot(node_counts, results_pdr, marker='d', color='navy', linestyle='-', linewidth=2)
    axs[0, 0].set_title('Packet Delivery Ratio (PDR)')
    axs[0, 0].set_xlabel('Number of WBANs')
    axs[0, 0].set_ylabel('PDR (%)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Average E2E Delay
    axs[0, 1].plot(node_counts, results_delay, marker='d', color='navy', linestyle='-', linewidth=2)
    axs[0, 1].set_title('Average E2E Delay')
    axs[0, 1].set_xlabel('Number of WBANs')
    axs[0, 1].set_ylabel('Delay (ms)')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Hop Count
    # Using bar charts to match paper styling (Figure 10)
    width = 40
    axs[1, 0].bar(np.array(node_counts), results_hops, width=width, color='dodgerblue', edgecolor='black')
    axs[1, 0].set_title('Average Hop Count')
    axs[1, 0].set_xlabel('Number of WBANs')
    axs[1, 0].set_ylabel('Hop Count')
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.6)

    # 4. Energy Consumption
    axs[1, 1].bar(np.array(node_counts), results_energy, width=width, color='dodgerblue', edgecolor='black')
    axs[1, 1].set_title('Total Energy Consumption')
    axs[1, 1].set_xlabel('Number of WBANs')
    axs[1, 1].set_ylabel('Energy (J)')
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying evaluation graphs...")
    plt.show()

if __name__ == "__main__":
    run_benchmark()
