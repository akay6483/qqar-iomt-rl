import random

class PlainQLearningAgent:
    def __init__(self, network_env, alpha=0.5, gamma=0.9, max_episodes=2000):
        self.env = network_env
        self.alpha = alpha
        self.gamma = gamma
        self.max_episodes = max_episodes

    def get_q_value(self, state_id, action_id):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'basic_q_table'): 
            node.basic_q_table = {}
        return node.basic_q_table.get(action_id, 0.0)

    def set_q_value(self, state_id, action_id, value):
        node = self.env.nodes[state_id]
        if not hasattr(node, 'basic_q_table'): 
            node.basic_q_table = {}
        node.basic_q_table[action_id] = value

    def train(self, sinks):
        """Standard 1-hop Q-learning ignoring QoS parameters."""
        print(f"Starting Plain Q-Learning training with Sinks: {sinks}")
        wban_nodes = [n for n in self.env.nodes.keys() if n not in sinks]
        
        for episode in range(self.max_episodes):
            current_id = random.choice(wban_nodes)
            steps = 0
            epsilon = max(0.1, 1.0 - episode / (self.max_episodes * 0.5))
            
            while current_id not in sinks and steps < 50:
                current_node = self.env.nodes[current_id]
                # Plain Q-learning considers ALL immediate neighbors, not just forward-progress ones
                candidates = list(current_node.neighbor_list.keys()) 
                
                if not candidates: break
                    
                if random.random() < epsilon:
                    action_id = random.choice(candidates)
                else:
                    action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                
                # Simple reward: 100 for sink, 0 for anything else
                reward = 100.0 if action_id in sinks else 0.0
                
                # Bellman Equation (Standard)
                if action_id in sinks:
                    max_next_q = 0.0
                else:
                    next_cands = list(self.env.nodes[action_id].neighbor_list.keys())
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_cands]) if next_cands else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                
                current_id = action_id
                steps += 1
        print("Training complete! Plain Q-Tables populated.")

    def select_best_route(self, current_id, candidates, sinks):
        """Simply picks the highest Q-value without any fallback congestion logic."""
        # Note: We pass candidates to keep the function signature identical to QQAR
        valid_candidates = [c for c in candidates if c in self.env.nodes[current_id].neighbor_list]
        if not valid_candidates: return None
        return max(valid_candidates, key=lambda c: self.get_q_value(current_id, c))