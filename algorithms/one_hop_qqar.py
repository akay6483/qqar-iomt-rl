import random

class OneHopQQARAgent:
    def __init__(self, network_env, alpha=0.5, gamma=0.9, max_episodes=2000):
        self.env = network_env
        self.alpha = alpha               
        self.gamma = gamma               
        self.max_episodes = max_episodes 
        self.weights = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}

    def get_q_value(self, state_id, action_id):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'one_hop_q_table'): node.one_hop_q_table = {}
        return node.one_hop_q_table.get(action_id, 0.0)

    def set_q_value(self, state_id, action_id, value):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'one_hop_q_table'): node.one_hop_q_table = {}
        node.one_hop_q_table[action_id] = value

    def get_distance_to_closest_sink(self, node_id, sinks):
        return min([self.env.get_distance(node_id, s) for s in sinks])

    def calculate_reward(self, current_id, next_hop_id, sinks, candidates):
        if next_hop_id in sinks: return 100.0  
            
        next_node = self.env.nodes[next_hop_id]
        dist_current = self.get_distance_to_closest_sink(current_id, sinks)
        dist_next = self.get_distance_to_closest_sink(next_hop_id, sinks)
        z_raw = max(0.01, dist_current - dist_next) 
        
        lr_raw = getattr(next_node, 'link_reliability', 1.0)
        e_raw = next_node.energy 
        d_raw = 1.0 / max(0.01, getattr(next_node, 'e2e_delay', 0.05)) 
        
        sum_lr, sum_z, sum_e, sum_d = 0.01, 0.01, 0.01, 0.01 
        for c_id in candidates:
            c_node = self.env.nodes[c_id]
            sum_lr += c_node.link_reliability
            sum_z += max(0.01, dist_current - self.get_distance_to_closest_sink(c_id, sinks))
            sum_e += c_node.energy
            sum_d += 1.0 / c_node.get_dynamic_delay() 
            
        r_val = (self.weights['A']*(lr_raw/sum_lr) + self.weights['B']*(z_raw/sum_z) + 
                 self.weights['C']*(d_raw/sum_d) + self.weights['D']*(e_raw/sum_e)) * 10
        return r_val

    def train(self, sinks):
        print(f"Starting 1-Hop QQAR training with Sinks: {sinks}")
        wban_nodes = [n for n in self.env.nodes.keys() if n not in sinks]
        
        for episode in range(self.max_episodes):
            current_id = random.choice(wban_nodes)
            steps = 0
            epsilon = max(0.1, 1.0 - episode / (self.max_episodes * 0.5))
            
            while current_id not in sinks and steps < 50:
                current_node = self.env.nodes[current_id]
                current_dist = self.get_distance_to_closest_sink(current_id, sinks)
                
                # STRICT FILTER: Only 1-hop neighbors allowed
                candidates = [n_id for n_id, data in current_node.neighbor_list.items() 
                              if data['hops'] == 1 and current_dist > self.get_distance_to_closest_sink(n_id, sinks)]
                              
                if not candidates: break
                    
                if random.random() < epsilon: action_id = random.choice(candidates)
                else: action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                    
                reward = self.calculate_reward(current_id, action_id, sinks, candidates)
                
                if action_id in sinks: max_next_q = 0.0
                else:
                    next_node = self.env.nodes[action_id]
                    next_dist = self.get_distance_to_closest_sink(action_id, sinks)
                    # STRICT FILTER for max_next_q: Only 1-hop neighbors of the next node
                    next_candidates = [n for n, data in next_node.neighbor_list.items() 
                                       if data['hops'] == 1 and next_dist > self.get_distance_to_closest_sink(n, sinks)]
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_candidates]) if next_candidates else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                
                current_id = action_id
                steps += 1
        print("Training complete! 1-Hop QQAR Q-Tables populated.")

    def select_best_route(self, current_id, candidates, sinks):
        # Filter physical candidates to 1-hop only
        one_hop_cands = [c for c in candidates if self.env.nodes[current_id].neighbor_list[c]['hops'] == 1]
        if not one_hop_cands: return None
        
        sorted_candidates = sorted(one_hop_cands, key=lambda c: self.get_q_value(current_id, c), reverse=True)
        best_candidate = sorted_candidates[0]
        best_node = self.env.nodes[best_candidate]
        traffic_load = getattr(best_node, 'avg_traffic_load', 0.0)
        
        if (traffic_load > 0.5 or best_node.energy < 20.0) and len(sorted_candidates) > 1:
            return sorted_candidates[1] 
        return best_candidate