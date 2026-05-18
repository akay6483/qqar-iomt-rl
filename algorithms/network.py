import networkx as nx
import random
import math
from algorithms.node import Node

class NetworkEnv:
    def __init__(self, area_size=500, num_nodes=200, tx_range=50):
        self.area_size = area_size
        self.num_nodes = num_nodes
        self.tx_range = tx_range
        
        self.nodes = {} # Dictionary mapping node_id -> Node object
        self.graph = nx.Graph() # The physical radio-frequency graph
        self.routing_overhead_bytes = 0
        self.control_energy_consumed = 0.0

    def record_control_packet(self, size_bytes, sender_id=None, receiver_ids=None):
        self.routing_overhead_bytes += size_bytes
        if sender_id is not None:
            self.control_energy_consumed += self.nodes[sender_id].consume_tx_energy(size_bytes)
        for receiver_id in receiver_ids or []:
            self.control_energy_consumed += self.nodes[receiver_id].consume_rx_energy(size_bytes)

    def reset_routing_overhead(self):
        self.routing_overhead_bytes = 0
        self.control_energy_consumed = 0.0

    def clone_topology(self):
        """
        Build a fresh environment with the same node positions and physical links.
        Dynamic routing tables, queues, energy, and Q-tables are intentionally reset.
        """
        clone = NetworkEnv(self.area_size, self.num_nodes, self.tx_range)
        clone.graph = self.graph.copy()
        for node_id in self.graph.nodes:
            node = Node(node_id=node_id)
            node.network_env = clone
            clone.nodes[node_id] = node
        return clone

    def deploy_nodes(self):
        """
        Scatters nodes randomly across the 500x500 grid and builds physical links.
        """
        print(f"Deploying {self.num_nodes} nodes in a {self.area_size}x{self.area_size}m area...")
        
        # 1. Create nodes and assign random (x, y) coordinates
        for i in range(self.num_nodes):
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            
            # Initialize our Node from node.py
            node = Node(node_id=i)
            node.network_env = self # Give the node a reference to the physical world
            
            self.nodes[i] = node
            self.graph.add_node(i, pos=(x, y))
            
        # 2. Build physical links based on the 50m transmission range
        positions = nx.get_node_attributes(self.graph, 'pos')
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # Calculate Euclidean distance
                dist = math.hypot(positions[i][0] - positions[j][0], 
                                  positions[i][1] - positions[j][1])
                
                # If they are within 50 meters, they can hear each other's radio
                if dist <= self.tx_range:
                    self.graph.add_edge(i, j, distance=dist)
                    
        print(f"Network built. Total physical links established: {self.graph.number_of_edges()}")

    def broadcast(self, sender_node, packet):
        """
        Simulates the physical transmission of a radio wave.
        Only nodes directly connected in the physical graph will receive the packet.
        """
        physical_neighbors = list(self.graph.neighbors(sender_node.node_id))
        self.record_control_packet(
            size_bytes=packet_size(packet),
            sender_id=sender_node.node_id,
            receiver_ids=physical_neighbors,
        )
        
        for neighbor_id in physical_neighbors:
            receiving_node = self.nodes[neighbor_id]
            receiving_node.receive_hello(packet)
            
    def get_distance(self, node_a_id, node_b_id):
        """Helper to calculate Euclidean distance between two nodes."""
        pos_a = self.graph.nodes[node_a_id]['pos']
        pos_b = self.graph.nodes[node_b_id]['pos']
        import math
        return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])


def packet_size(packet):
    from algorithms.paper_config import HELLO_PACKET_BYTES

    return HELLO_PACKET_BYTES
