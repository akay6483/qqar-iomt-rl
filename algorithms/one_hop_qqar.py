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
    Q_UPDATE_PACKET_BYTES_QQAR,
    TRAFFIC_LOAD_THRESHOLD,
    LABEL_QQAR_1HOP,
)

class OneHopQQARAgent:
    def __init__(self, network_env, alpha=ALPHA, gamma=GAMMA, max_episodes=MAX_EPISODES):
        self.env = network_env
        self.alpha = alpha               
        self.gamma = gamma               
        self.max_episodes = max_episodes 
        self.weights = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
        self.routing_overhead_bytes = 0
        self.training_energy_consumed = 0.0
        self.training_energy_history = []

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

    def calculate_reward(self, current_id, next_hop_id, sinks, candidates, packet):
        if next_hop_id in sinks: return 100.0  
            
        next_node = self.env.nodes[next_hop_id]
        
        # Calculate distance to closest sink
        dist_current = min([self.env.get_distance(current_id, s) for s in sinks])
        dist_next = min([self.env.get_distance(next_hop_id, s) for s in sinks])
        
        deadline_remaining = max(0.01, packet.absolute_deadline - packet.creation_time)
        z_raw = max(0.01, (dist_current - dist_next) / deadline_remaining) 
        
        lr_raw = self.env.nodes[current_id].get_link_reliability(next_hop_id)
        e_raw = max(0.01, next_node.energy)
        observed_delay = next_node.get_dynamic_delay() + getattr(next_node, 'avg_traffic_load', 0.0) * 0.05
        d_raw = 1.0 / max(0.01, observed_delay)
        
        sum_lr, sum_z, sum_e, sum_d = 0.01, 0.01, 0.01, 0.01 
        for c_id in candidates:
            c_node = self.env.nodes[c_id]
            sum_lr += self.env.nodes[current_id].get_link_reliability(c_id)
            
            # Use closest sink in sum_z normalization
            c_dist = min([self.env.get_distance(c_id, s) for s in sinks])
            sum_z += max(0.01, (dist_current - c_dist) / deadline_remaining)
            
            sum_e += max(0.01, c_node.energy)
            c_delay = c_node.get_dynamic_delay() + getattr(c_node, 'avg_traffic_load', 0.0) * 0.05
            sum_d += 1.0 / max(0.01, c_delay)
            
        r_val = (self.weights['A']*(lr_raw/sum_lr) + self.weights['B']*(z_raw/sum_z) + 
                 self.weights['C']*(d_raw/sum_d) + self.weights['D']*(e_raw/sum_e)) * 100
        return r_val

    def _record_q_update_overhead(self, current_id, next_hop_id):
        self.routing_overhead_bytes += Q_UPDATE_PACKET_BYTES_QQAR
        self.training_energy_consumed += self.env.nodes[current_id].consume_tx_energy(Q_UPDATE_PACKET_BYTES_QQAR)
        self.training_energy_consumed += self.env.nodes[next_hop_id].consume_rx_energy(Q_UPDATE_PACKET_BYTES_QQAR)

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
        print(f"Training {LABEL_QQAR_1HOP}: sinks={len(sinks)}, episodes={self.max_episodes}")
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
                
                candidates = [n_id for n_id, data in current_node.neighbor_list.items() 
                              if data['hops'] == 1
                              and current_dist > min([self.env.get_distance(n_id, s) for s in sinks])]
                              
                if not candidates: break
                    
                if random.random() < epsilon: action_id = random.choice(candidates)
                else: action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                    
                success = self._transmission_succeeds(current_id, action_id)
                
                reward = self.calculate_reward(current_id, action_id, sinks, candidates, pkt)
                if not success: reward = -100.0
                
                if action_id in sinks: max_next_q = 0.0
                else:
                    next_node = self.env.nodes[action_id]
                    next_dist = min([self.env.get_distance(action_id, s) for s in sinks])
                    next_candidates = [n for n, data in next_node.neighbor_list.items() 
                                       if data['hops'] == 1
                                       and next_dist > min([self.env.get_distance(n, s) for s in sinks])]
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_candidates]) if next_candidates else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                self._record_q_update_overhead(current_id, action_id)
                self._apply_training_transition(current_id, action_id, success)
                
                if not success:
                    break
                current_id = action_id
                steps += 1
            self.training_energy_history.append(self.training_energy_consumed)
        print(f"Finished {LABEL_QQAR_1HOP}.")

    def select_best_route(self, current_id, candidates, sinks):
        # Filter physical candidates to 1-hop only
        one_hop_cands = [c for c in candidates if self.env.nodes[current_id].neighbor_list[c]['hops'] == 1]
        if not one_hop_cands: return None
        
        sorted_candidates = sorted(one_hop_cands, key=lambda c: self.get_q_value(current_id, c), reverse=True)
        best_candidate = sorted_candidates[0]
        best_node = self.env.nodes[best_candidate]
        traffic_load = getattr(best_node, 'avg_traffic_load', 0.0)
        
        if (traffic_load > TRAFFIC_LOAD_THRESHOLD or best_node.energy < LOW_ENERGY_THRESHOLD_J) and len(sorted_candidates) > 1:
            return sorted_candidates[1] 
        return best_candidate
