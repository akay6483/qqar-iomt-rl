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

    def calculate_reward(self, current_id, next_hop_id, sink_id, candidates, packet):
        """Calculates the normalized multi-parameter reward based on Eq. 25."""
        if next_hop_id == sink_id:
            return 100.0  # r_max: Reached ANY destination sink
            
        next_node = self.env.nodes[next_hop_id]
        
        # Calculate raw values for the chosen next hop
        dist_current = self.env.get_distance(current_id, sink_id)
        dist_next = self.env.get_distance(next_hop_id, sink_id)
        deadline_remaining = max(0.01, packet.absolute_deadline - packet.creation_time)
        z_raw = max(0.01, (dist_current - dist_next) / deadline_remaining) # Velocity (Progress)
        
        lr_raw = self.env.nodes[current_id].get_link_reliability(next_hop_id)
        e_raw = max(0.01, next_node.energy) # Using residual energy to maximize reward
        observed_delay = next_node.get_dynamic_delay() + getattr(next_node, 'avg_traffic_load', 0.0) * 0.05
        d_raw = 1.0 / max(0.01, observed_delay) # Inverse delay
        
        # Calculate sums across all candidates for normalization (Eq. 25 denominator)
        sum_lr, sum_z, sum_e, sum_d = 0.01, 0.01, 0.01, 0.01 # Prevent div by zero
        for c_id in candidates:
            c_node = self.env.nodes[c_id]
            sum_lr += self.env.nodes[current_id].get_link_reliability(c_id)
            c_dist = self.env.get_distance(c_id, sink_id)
            sum_z += max(0.01, (dist_current - c_dist) / deadline_remaining)
            sum_e += max(0.01, c_node.energy)
            c_delay = c_node.get_dynamic_delay() + getattr(c_node, 'avg_traffic_load', 0.0) * 0.05
            sum_d += 1.0 / max(0.01, c_delay) # Dynamic
            
        # Eq 25: Normalized multi-parameter reward
        r_val = (self.weights['A'] * (lr_raw / sum_lr) + 
                 self.weights['B'] * (z_raw / sum_z) + 
                 self.weights['C'] * (d_raw / sum_d) + 
                 self.weights['D'] * (e_raw / sum_e)) * 100
                 
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
        print(f"[Train] {LABEL_QQAR_2HOP}: sink_count={len(sinks)}, episodes={self.max_episodes}")
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
            
            while current_id != pkt.sink_id and steps < MAX_TRAINING_STEPS:
                current_node = self.env.nodes[current_id]
                
                # Filter candidates: Only pick neighbors that bring us closer to the nearest sink
                current_dist = self.env.get_distance(current_id, pkt.sink_id)
                candidates = [n_id for n_id in current_node.neighbor_list.keys() 
                              if (n_id not in sinks or n_id == pkt.sink_id)
                              and current_dist > self.env.get_distance(n_id, pkt.sink_id)]
                              
                if not candidates:
                    break # Trapped in a local minimum
                    
                # Action Selection
                if random.random() < epsilon:
                    action_id = random.choice(candidates)
                else:
                    action_id = max(candidates, key=lambda a: self.get_q_value(current_id, a))
                    
                success = self._apply_training_route(current_id, action_id)
                reward = self.calculate_reward(current_id, action_id, pkt.sink_id, candidates, pkt)
                if not success:
                    reward = -100.0
                
                # Bellman Equation
                if action_id == pkt.sink_id:
                    max_next_q = 0.0
                else:
                    next_node = self.env.nodes[action_id]
                    next_dist = self.env.get_distance(action_id, pkt.sink_id)
                    next_candidates = [n for n in next_node.neighbor_list.keys() 
                                       if (n not in sinks or n == pkt.sink_id)
                                       and next_dist > self.env.get_distance(n, pkt.sink_id)]
                    
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
                
        print(f"[Train] {LABEL_QQAR_2HOP}: complete; Q-tables populated.")

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
        
        if (traffic_load > TRAFFIC_LOAD_THRESHOLD or best_node.energy < LOW_ENERGY_THRESHOLD_J) and len(sorted_candidates) > 1:
            return sorted_candidates[1] # Use the alternative path
            
        return best_candidate
