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
    LABEL_QQAR_2HOP,
)

class QLearningAgent:
    def __init__(self, network_env, alpha=ALPHA, gamma=GAMMA, max_episodes=MAX_EPISODES):
        self.env = network_env
        self.alpha = alpha               
        self.gamma = gamma               
        self.max_episodes = max_episodes 
        
        # Reward weights (A, B, C, D must sum to 1.0)
        self.weights = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
        self.routing_overhead_bytes = 0
        self.training_energy_consumed = 0.0
        self.training_energy_history = []

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
        return min([self.env.get_distance(node_id, s) for s in sinks])

    def calculate_reward(self, current_id, next_hop_id, sinks, candidates, packet):
        if next_hop_id in sinks: return 100.0  
            
        next_node = self.env.nodes[next_hop_id]
        neighbor_info = self.env.nodes[current_id].neighbor_list.get(next_hop_id, {})
        
        # Calculate distance to closest sink
        dist_current = min([self.env.get_distance(current_id, s) for s in sinks])
        dist_next = min([self.env.get_distance(next_hop_id, s) for s in sinks])
        deadline_remaining = max(0.01, packet.absolute_deadline - packet.creation_time)
        z_raw = max(0.01, (dist_current - dist_next) / deadline_remaining) 
        
        # BUG FIX: If it's a 2-hop route, evaluate the RELAY node's reliability and energy
        if neighbor_info.get('hops') == 2:
            relay_id = neighbor_info['relay']
            relay_node = self.env.nodes[relay_id]
            lr_raw = self.env.nodes[current_id].get_link_reliability(relay_id) * relay_node.get_link_reliability(next_hop_id)
            e_raw = min(max(0.01, next_node.energy), max(0.01, relay_node.energy)) # Bottleneck energy
            observed_delay = relay_node.get_dynamic_delay() + getattr(relay_node, 'avg_traffic_load', 0.0) * 0.05
        else:
            lr_raw = self.env.nodes[current_id].get_link_reliability(next_hop_id)
            e_raw = max(0.01, next_node.energy)
            observed_delay = next_node.get_dynamic_delay() + getattr(next_node, 'avg_traffic_load', 0.0) * 0.05
            
        d_raw = 1.0 / max(0.01, observed_delay)
        
        # Normalization sums
        sum_lr, sum_z, sum_e, sum_d = 0.01, 0.01, 0.01, 0.01 
        for c_id in candidates:
            c_info = self.env.nodes[current_id].neighbor_list.get(c_id, {})
            c_node = self.env.nodes[c_id]
            
            if c_info.get('hops') == 2:
                c_relay = self.env.nodes[c_info['relay']]
                sum_lr += self.env.nodes[current_id].get_link_reliability(c_relay.node_id) * c_relay.get_link_reliability(c_id)
                sum_e += min(max(0.01, c_node.energy), max(0.01, c_relay.energy))
                c_delay = c_relay.get_dynamic_delay() + getattr(c_relay, 'avg_traffic_load', 0.0) * 0.05
            else:
                sum_lr += self.env.nodes[current_id].get_link_reliability(c_id)
                sum_e += max(0.01, c_node.energy)
                c_delay = c_node.get_dynamic_delay() + getattr(c_node, 'avg_traffic_load', 0.0) * 0.05
                
            c_dist = min([self.env.get_distance(c_id, s) for s in sinks])
            sum_z += max(0.01, (dist_current - c_dist) / deadline_remaining)
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

    def _route_hops(self, current_id, selected_next):
        neighbor_info = self.env.nodes[current_id].neighbor_list.get(selected_next)
        if not neighbor_info:
            return []

        if neighbor_info['hops'] == 1:
            return [selected_next]

        relay_id = neighbor_info.get('relay')
        if relay_id is None:
            return []
        if not self.env.graph.has_edge(current_id, relay_id):
            return []
        if not self.env.graph.has_edge(relay_id, selected_next):
            return []
        return [relay_id, selected_next]

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

    def _apply_training_route(self, current_id, selected_next):
        route_hops = self._route_hops(current_id, selected_next)
        if not route_hops:
            return False

        transmitter_id = current_id
        for hop_id in route_hops:
            success = self._transmission_succeeds(transmitter_id, hop_id)
            self._apply_training_transition(transmitter_id, hop_id, success)
            if not success:
                return False
            transmitter_id = hop_id
        return True

    def train(self, sinks):
        """
        Decentralized training: Spawns random packets from random nodes 
        to naturally build Q-tables across the entire network.
        """
        print(f"Training {LABEL_QQAR_2HOP}: sinks={len(sinks)}, episodes={self.max_episodes}")
        self.training_energy_consumed = 0.0
        self.training_energy_history = []
        
        # Get list of all WBAN nodes (excluding sinks)
        wban_nodes = [n for n in self.env.nodes.keys() if n not in sinks]
        
        for episode in range(self.max_episodes):
            # Pick a random source node for this episode
            current_id = random.choice(wban_nodes)
            pkt = self._sample_training_packet(episode + 1, current_id, sinks)
            steps = 0
            
            epsilon = max(0.1, 1.0 - episode / (self.max_episodes * 0.5))
            
            while current_id not in sinks and steps < MAX_TRAINING_STEPS:
                current_node = self.env.nodes[current_id]
                current_dist = min([self.env.get_distance(current_id, s) for s in sinks])
                
                candidates = [n_id for n_id, data in current_node.neighbor_list.items() 
                              if current_dist > min([self.env.get_distance(n_id, s) for s in sinks])]
                              
                if not candidates: break
                    
                if random.random() < epsilon: action_id = random.choice(candidates)
                else: action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                    
                success = self._apply_training_route(current_id, action_id)
                
                reward = self.calculate_reward(current_id, action_id, sinks, candidates, pkt)
                if not success: reward = -100.0
                
                if action_id in sinks: max_next_q = 0.0
                else:
                    next_node = self.env.nodes[action_id]
                    next_dist = min([self.env.get_distance(action_id, s) for s in sinks])
                    next_candidates = [n for n, data in next_node.neighbor_list.items() 
                                       if next_dist > min([self.env.get_distance(n, s) for s in sinks])]
                    max_next_q = max([self.get_q_value(action_id, a) for a in next_candidates]) if next_candidates else 0.0
                    
                old_q = self.get_q_value(current_id, action_id)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q_value(current_id, action_id, new_q)
                self._record_q_update_overhead(current_id, action_id)

                if not success:
                    break
                current_id = action_id
                steps += 1

            self.training_energy_history.append(self.training_energy_consumed)
                
        print(f"Finished {LABEL_QQAR_2HOP}.")

    def select_best_route(self, current_id, candidates, sinks):
        """
        Algorithm 4 (Lines 26-28): Fallback congestion control.
        """
        sorted_candidates = sorted(candidates, key=lambda c: self.get_q_value(current_id, c), reverse=True)
        
        # BUG FIX: Loop through best candidates and check the RELAY node if it's a 2-hop route
        for best_candidate in sorted_candidates:
            best_node = self.env.nodes[best_candidate]
            neighbor_info = self.env.nodes[current_id].neighbor_list.get(best_candidate, {})
            
            if neighbor_info.get('hops') == 2:
                check_node = self.env.nodes[neighbor_info['relay']] # Check the middle-man
            else:
                check_node = best_node # Check the direct receiver
                
            traffic_load = getattr(check_node, 'avg_traffic_load', 0.0)
            
            # If the node we are handing the packet to is healthy, take this route
            if traffic_load <= TRAFFIC_LOAD_THRESHOLD and check_node.energy >= LOW_ENERGY_THRESHOLD_J:
                return best_candidate
                
        # If all preferred routes are congested, default to the highest Q-value
        return sorted_candidates[0] if sorted_candidates else None
