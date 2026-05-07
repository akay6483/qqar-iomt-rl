import random

class QLearningAgent:
    def __init__(self, network_env, alpha=0.5, gamma=0.9, max_episodes=5000):
        self.env = network_env
        self.alpha = alpha               
        self.gamma = gamma               
        self.max_episodes = max_episodes 
        
        # Reward weights (A, B, C, D must sum to 1.0)
        self.weights = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}

    def get_q_value(self, state_id, action_id):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'q_table'):
            node.q_table = {}
        return node.q_table.get(action_id, 0.0)

    def set_q_value(self, state_id, action_id, value):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'q_table'):
            node.q_table = {}
        node.q_table[action_id] = value

    def get_distance_to_closest_sink(self, node_id, sinks):
        """Finds the Euclidean distance to the nearest sink node."""
        return min([self.env.get_distance(node_id, s) for s in sinks])

    def calculate_reward(self, current_id, next_hop_id, sinks, candidates):
        """Calculates the normalized multi-parameter reward based on Eq. 25."""
        if next_hop_id in sinks:
            return 100.0  # r_max: Reached ANY destination sink
            
        next_node = self.env.nodes[next_hop_id]
        
        # Calculate raw values for the chosen next hop
        dist_current = self.get_distance_to_closest_sink(current_id, sinks)
        dist_next = self.get_distance_to_closest_sink(next_hop_id, sinks)
        z_raw = max(0.01, dist_current - dist_next) # Velocity (Progress)
        
        lr_raw = getattr(next_node, 'link_reliability', 1.0)
        e_raw = next_node.energy # Using residual energy to maximize reward
        d_raw = 1.0 / max(0.01, getattr(next_node, 'e2e_delay', 0.05)) # Inverse delay
        
        # Calculate sums across all candidates for normalization (Eq. 25 denominator)
        sum_lr, sum_z, sum_e, sum_d = 0.01, 0.01, 0.01, 0.01 # Prevent div by zero
        for c_id in candidates:
            c_node = self.env.nodes[c_id]
            sum_lr += c_node.link_reliability
            c_dist = self.get_distance_to_closest_sink(c_id, sinks)
            sum_z += max(0.01, dist_current - c_dist)
            sum_e += c_node.energy
            sum_d += 1.0 / c_node.get_dynamic_delay() # Dynamic
            
        # Eq 25: Normalized multi-parameter reward
        r_val = (self.weights['A'] * (lr_raw / sum_lr) + 
                 self.weights['B'] * (z_raw / sum_z) + 
                 self.weights['C'] * (d_raw / sum_d) + 
                 self.weights['D'] * (e_raw / sum_e)) * 10
                 
        return r_val

    def train(self, sinks):
        """
        Decentralized training: Spawns random packets from random nodes 
        to naturally build Q-tables across the entire network.
        """
        print(f"Starting Q-Learning training with Sinks: {sinks}")
        
        # Get list of all WBAN nodes (excluding sinks)
        wban_nodes = [n for n in self.env.nodes.keys() if n not in sinks]
        
        for episode in range(self.max_episodes):
            # Pick a random source node for this episode
            current_id = random.choice(wban_nodes)
            steps = 0
            
            epsilon = max(0.1, 1.0 - episode / (self.max_episodes * 0.5))
            
            while current_id not in sinks and steps < 50:
                current_node = self.env.nodes[current_id]
                
                # Filter candidates: Only pick neighbors that bring us closer to the nearest sink
                current_dist = self.get_distance_to_closest_sink(current_id, sinks)
                candidates = [n_id for n_id in current_node.neighbor_list.keys() 
                              if current_dist > self.get_distance_to_closest_sink(n_id, sinks)]
                              
                if not candidates:
                    break # Trapped in a local minimum
                    
                # Action Selection
                if random.random() < epsilon:
                    action_id = random.choice(candidates)
                else:
                    action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                    
                reward = self.calculate_reward(current_id, action_id, sinks, candidates)
                
                # Bellman Equation
                if action_id in sinks:
                    max_next_q = 0.0
                else:
                    next_node = self.env.nodes[action_id]
                    next_dist = self.get_distance_to_closest_sink(action_id, sinks)
                    next_candidates = [n for n in next_node.neighbor_list.keys() 
                                       if next_dist > self.get_distance_to_closest_sink(n, sinks)]
                    
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_candidates]) if next_candidates else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                
                current_id = action_id
                steps += 1
                
        print("Training complete! Network-wide Q-Tables populated.")

    def select_best_route(self, current_id, candidates, sinks):
        """
        Algorithm 4 (Lines 26-28): Fallback congestion control.
        """
        sorted_candidates = sorted(candidates, key=lambda c: self.get_q_value(current_id, c), reverse=True)
        best_candidate = sorted_candidates[0]
        best_node = self.env.nodes[best_candidate]
        
        # Assume nodes have a traffic load tracker. Fallback if overloaded or dying.
        # L_th is 0.5 as defined in the paper.
        traffic_load = getattr(best_node, 'avg_traffic_load', 0.0)
        
        if (traffic_load > 0.5 or best_node.energy < 20.0) and len(sorted_candidates) > 1:
            return sorted_candidates[1] # Use the alternative path
            
        return best_candidate
