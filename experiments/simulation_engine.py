import random
from algorithms.node import DataPacket, PriorityScheduler
from algorithms.paper_config import (
    BANDWIDTH_BPS,
    INITIAL_ENERGY_J,
    LINK_FEEDBACK_PACKET_BYTES,
    PACKET_SIZE_BYTES,
    PRIORITY_PROFILE,
    TIME_STEP_SECONDS,
)

class SimulationEngine:
    def __init__(
        self,
        env,
        agent,
        sinks,
        wban_nodes,
        data_rate,
        max_time=15.0,
        traffic_seed=None,
        channel_seed=None,
        initial_energy=INITIAL_ENERGY_J,
    ):
        self.env = env
        self.agent = agent
        self.sinks = sinks
        self.wban_nodes = wban_nodes
        self.data_rate = data_rate
        self.max_time = max_time
        self.current_time = 0.0
        self.time_step = TIME_STEP_SECONDS
        self.traffic_rng = random.Random(traffic_seed) if traffic_seed is not None else random
        self.channel_rng = random.Random(channel_seed) if channel_seed is not None else random
        self.routing_overhead_bytes = 0
        
        for node in self.env.nodes.values():
            node.scheduler = PriorityScheduler()
            node.pkt_in = 0
            node.pkt_out = 0
            node.total_tx = 0
            node.success_tx = 0
            node.link_reliability = 1.0
            node.link_stats = {}
            node.avg_traffic_load = 0.0
            if isinstance(initial_energy, dict):
                node.energy = initial_energy[node.node_id]
            elif initial_energy is not None:
                node.energy = initial_energy

        self.successful_packets = 0
        self.generated_packets = 0
        self.dropped_packets = 0
        self.expired_packets = 0
        self.failed_packets = 0
        self.total_delay = 0.0
        self.total_hops = 0
        self.energy_consumed = 0.0
        self.active_packets = {}

    def generate_traffic(self):
        # BOTTLENECK FIXED: Accurately spawn bulk traffic for heavy density
        expected_packets = (self.data_rate * len(self.wban_nodes)) * self.time_step
        num_packets = int(expected_packets)
        
        if self.traffic_rng.random() < (expected_packets - num_packets):
            num_packets += 1
            
        for _ in range(num_packets):
            self.generated_packets += 1
            src = self.traffic_rng.choice(self.wban_nodes)
            snk = self.traffic_rng.choice(self.sinks)
            
            rv = self.traffic_rng.random()
            for cutoff, tag, dl in PRIORITY_PROFILE:
                if rv <= cutoff:
                    break
                
            pkt = DataPacket(self.generated_packets, src, snk, tag, self.current_time, dl)
            self.active_packets[pkt.packet_id] = {'creation': self.current_time, 'hops': 0}
            if not self.env.nodes[src].scheduler.enqueue_packet(pkt):
                self.dropped_packets += 1
                self.active_packets.pop(pkt.packet_id, None)

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

    def _transmit_hop(self, transmitter_id, receiver_id):
        transmitter = self.env.nodes[transmitter_id]
        receiver = self.env.nodes[receiver_id]
        if transmitter.energy <= 0:
            return False, 0.0

        # Per-hop routing feedback used to maintain link reliability/load metrics.
        self.routing_overhead_bytes += LINK_FEEDBACK_PACKET_BYTES
        self.energy_consumed += transmitter.consume_tx_energy()

        link_loss = 1.0 - min(1.0, transmitter.get_link_reliability(receiver_id))
        collision_chance = min(0.6, transmitter.avg_traffic_load * 0.15 + link_loss * 0.1)
        success = self.channel_rng.random() > collision_chance
        transmitter.record_transmission(success=success, neighbor_id=receiver_id)
        hop_delay = (PACKET_SIZE_BYTES * 8) / BANDWIDTH_BPS + transmitter.get_dynamic_delay()
        if success:
            self.energy_consumed += receiver.consume_rx_energy()
            self.energy_consumed += receiver.consume_tx_energy(LINK_FEEDBACK_PACKET_BYTES)
            self.energy_consumed += transmitter.consume_rx_energy(LINK_FEEDBACK_PACKET_BYTES)
            receiver.pkt_in += 1
        return success, hop_delay

    def _packet_elapsed_time(self, pkt):
        packet_state = self.active_packets[pkt.packet_id]
        return (
            self.current_time
            - pkt.creation_time
            + packet_state.get('last_delay', 0.0)
        )

    def _drop_expired_packet(self, pkt):
        self.expired_packets += 1
        self.active_packets.pop(pkt.packet_id, None)

    def run(self):
        while self.current_time < self.max_time:
            self.generate_traffic()
            for node_id, node in self.env.nodes.items():
                if node_id in self.sinks: continue
                
                node.scheduler.manage_timeouts(self.current_time, self.time_step)
                pkt = node.scheduler.get_next_packet_for_transmission()
                
                if pkt:
                    if self._packet_elapsed_time(pkt) >= (pkt.absolute_deadline - pkt.creation_time):
                        self._drop_expired_packet(pkt)
                        continue
                    
                    cur_dist = self.env.get_distance(node_id, pkt.sink_id)
                    candidates = [
                        n for n in node.neighbor_list.keys()
                        if (n not in self.sinks or n == pkt.sink_id)
                        and cur_dist > self.env.get_distance(n, pkt.sink_id)
                    ]
                    
                    if not candidates:
                        self.dropped_packets += 1
                        self.active_packets.pop(pkt.packet_id, None)
                        continue
                    
                    nxt = self.agent.select_best_route(node_id, candidates, [pkt.sink_id])
                    if not nxt:
                        self.dropped_packets += 1
                        self.active_packets.pop(pkt.packet_id, None)
                        continue
                    route_hops = self._route_hops(node_id, nxt)
                    if not route_hops:
                        self.dropped_packets += 1
                        self.active_packets.pop(pkt.packet_id, None)
                        continue
                    
                    delivered = False
                    failed = False
                    transmitter_id = node_id
                    final_node = node_id

                    for hop_id in route_hops:
                        if self._packet_elapsed_time(pkt) >= (pkt.absolute_deadline - pkt.creation_time):
                            failed = True
                            self._drop_expired_packet(pkt)
                            break

                        self.active_packets[pkt.packet_id]['hops'] += 1
                        success, hop_delay = self._transmit_hop(transmitter_id, hop_id)
                        self.active_packets[pkt.packet_id]['last_delay'] = (
                            self.active_packets[pkt.packet_id].get('last_delay', 0.0) + hop_delay
                        )
                        if not success:
                            failed = True
                            self.failed_packets += 1
                            self.active_packets.pop(pkt.packet_id, None)
                            break

                        if self._packet_elapsed_time(pkt) > (pkt.absolute_deadline - pkt.creation_time):
                            failed = True
                            self._drop_expired_packet(pkt)
                            break

                        final_node = hop_id
                        if hop_id == pkt.sink_id:
                            self.successful_packets += 1
                            self.total_hops += self.active_packets[pkt.packet_id]['hops']
                            queued_delay = self.current_time - pkt.creation_time
                            hop_delay_total = self.active_packets[pkt.packet_id].get('last_delay', 0.0)
                            self.total_delay += (queued_delay + hop_delay_total) * 1000
                            self.active_packets.pop(pkt.packet_id, None)
                            delivered = True
                            break
                        transmitter_id = hop_id

                    if not delivered and not failed and final_node not in self.sinks:
                        if not self.env.nodes[final_node].scheduler.enqueue_packet(pkt):
                            self.dropped_packets += 1
                            self.active_packets.pop(pkt.packet_id, None)
                        
            self.current_time += self.time_step
            
        return self._metrics()

    def _metrics(self):
        pdr = (self.successful_packets / max(1, self.generated_packets)) * 100
        return pdr, self.total_delay/max(1,self.successful_packets), self.total_hops/max(1,self.successful_packets), self.energy_consumed
