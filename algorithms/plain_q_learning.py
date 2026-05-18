import random
from algorithms.node import DataPacket
from algorithms.paper_config import (
    ALPHA,
    GAMMA,
    LINK_FEEDBACK_PACKET_BYTES,
    LOW_ENERGY_THRESHOLD_J,
    MAX_EPISODES,
    MAX_TRAINING_STEPS,
    PRIORITY_PROFILE,
    Q_UPDATE_PACKET_BYTES_PLAIN,
    LABEL_BASELINE_Q,
)

class PlainQLearningAgent:
    def __init__(self, network_env, alpha=ALPHA, gamma=GAMMA, max_episodes=MAX_EPISODES):
        self.env = network_env
        self.alpha = alpha
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.routing_overhead_bytes = 0
        self.training_energy_consumed = 0.0
        self.training_energy_history = []

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

    def _sample_training_packet(self, packet_id, source_id, sinks):
        rv = random.random()
        for cutoff, tag, deadline in PRIORITY_PROFILE:
            if rv <= cutoff:
                return DataPacket(packet_id, source_id, random.choice(sinks), tag, 0.0, deadline)
        cutoff, tag, deadline = PRIORITY_PROFILE[-1]
        return DataPacket(packet_id, source_id, random.choice(sinks), tag, 0.0, deadline)

    def _transmission_succeeds(self, current_id, next_hop_id):
        current_node = self.env.nodes[current_id]
        next_node = self.env.nodes[next_hop_id]
        if current_node.energy <= 0:
            return False

        congestion = min(0.6, getattr(current_node, 'avg_traffic_load', 0.0) * 0.15)
        weak_link = 1.0 - min(1.0, current_node.get_link_reliability(next_hop_id))
        low_energy = 0.25 if current_node.energy < LOW_ENERGY_THRESHOLD_J or next_node.energy < LOW_ENERGY_THRESHOLD_J else 0.0
        failure_probability = min(0.85, congestion + weak_link * 0.25 + low_energy)
        return random.random() > failure_probability

    def _apply_training_transition(self, current_id, next_hop_id, success):
        current_node = self.env.nodes[current_id]
        next_node = self.env.nodes[next_hop_id]

        self.routing_overhead_bytes += LINK_FEEDBACK_PACKET_BYTES
        self.training_energy_consumed += current_node.consume_tx_energy()
        if success:
            self.training_energy_consumed += next_node.consume_rx_energy()
            self.training_energy_consumed += next_node.consume_tx_energy(LINK_FEEDBACK_PACKET_BYTES)
            self.training_energy_consumed += current_node.consume_rx_energy(LINK_FEEDBACK_PACKET_BYTES)
            next_node.pkt_in += 1
        current_node.record_transmission(success=success, neighbor_id=next_hop_id)

        background_load = random.random() * 0.4
        current_node.avg_traffic_load = min(1.0, 0.7 * current_node.avg_traffic_load + 0.3 * background_load)
        next_node.avg_traffic_load = min(1.0, 0.8 * next_node.avg_traffic_load + 0.2 * background_load)
        current_node.e2e_delay = current_node.get_dynamic_delay() + current_node.avg_traffic_load * 0.05
        next_node.e2e_delay = next_node.get_dynamic_delay() + next_node.avg_traffic_load * 0.05

    def train(self, sinks):
        """Standard 1-hop Q-learning ignoring QoS parameters."""
        print(f"Training {LABEL_BASELINE_Q}: sinks={len(sinks)}, episodes={self.max_episodes}")
        self.training_energy_consumed = 0.0
        self.training_energy_history = []
        wban_nodes = [n for n in self.env.nodes.keys() if n not in sinks]
        
        for episode in range(self.max_episodes):
            current_id = random.choice(wban_nodes)
            pkt = self._sample_training_packet(episode + 1, current_id, sinks)
            steps = 0
            epsilon = max(0.1, 1.0 - episode / (self.max_episodes * 0.5))
            
            while current_id not in sinks and steps < MAX_TRAINING_STEPS:
                current_node = self.env.nodes[current_id]
                
                current_dist = min([self.env.get_distance(current_id, s) for s in sinks])
                candidates = [
                    n_id for n_id, data in current_node.neighbor_list.items()
                    if data['hops'] == 1
                    and current_dist > min([self.env.get_distance(n_id, s) for s in sinks])
                ]
                
                if not candidates: break
                    
                if random.random() < epsilon:
                    action_id = random.choice(candidates)
                else:
                    action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                
                success = self._transmission_succeeds(current_id, action_id)
                reward = 100.0 if action_id in sinks else 0.0
                if not success: reward = -100.0
                
                if action_id in sinks:
                    max_next_q = 0.0
                else:
                    next_cands = [
                        n for n, data in self.env.nodes[action_id].neighbor_list.items()
                        if data['hops'] == 1
                        and min([self.env.get_distance(action_id, s) for s in sinks]) > min([self.env.get_distance(n, s) for s in sinks])
                    ]
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_cands]) if next_cands else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                self.routing_overhead_bytes += Q_UPDATE_PACKET_BYTES_PLAIN
                self.training_energy_consumed += current_node.consume_tx_energy(Q_UPDATE_PACKET_BYTES_PLAIN)
                self.training_energy_consumed += self.env.nodes[action_id].consume_rx_energy(Q_UPDATE_PACKET_BYTES_PLAIN)
                self._apply_training_transition(current_id, action_id, success)
                
                if not success:
                    break
                current_id = action_id
                steps += 1
            self.training_energy_history.append(self.training_energy_consumed)
        print(f"Finished {LABEL_BASELINE_Q}.")

    def select_best_route(self, current_id, candidates, sinks):
        """Simply picks the highest Q-value without any fallback congestion logic."""
        # Note: We pass candidates to keep the function signature identical to QQAR
        valid_candidates = [
            c for c in candidates
            if c in self.env.nodes[current_id].neighbor_list
            and self.env.nodes[current_id].neighbor_list[c]['hops'] == 1
        ]
        if not valid_candidates: return None
        return max(valid_candidates, key=lambda c: self.get_q_value(current_id, c))
