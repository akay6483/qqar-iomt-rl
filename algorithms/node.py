from algorithms.paper_config import (
    HELLO_PACKET_BYTES,
    PACKET_SIZE_BYTES,
    QUEUE_CAPACITY,
    RX_ENERGY_PER_BYTE_J,
    TX_ENERGY_PER_BYTE_J,
)

class HelloPacket:
    def __init__(self, source_id, relay_id, time_slot, energy_level):
        self.sn = source_id
        self.rn = relay_id
        self.t_a = time_slot
        self.energy = energy_level

class DataPacket:
    def __init__(self, packet_id, source_id, sink_id, priority_tag, creation_time, deadline_duration):
        self.packet_id = packet_id
        self.source_id = source_id
        self.sink_id = sink_id
        self.priority_tag = priority_tag
        self.creation_time = creation_time
        self.absolute_deadline = creation_time + deadline_duration

class PriorityScheduler:
    def __init__(self):
        self.critical_queue = []
        self.delay_sensitive_queue = []
        self.reliability_sensitive_queue = []
        self.ordinary_queue = []

    def enqueue_packet(self, packet):
        if self.size() >= QUEUE_CAPACITY:
            return False
        if packet.priority_tag == 'High+': self.critical_queue.append(packet)
        elif packet.priority_tag == 'High': self.delay_sensitive_queue.append(packet)
        elif packet.priority_tag == 'Med': self.reliability_sensitive_queue.append(packet)
        else: self.ordinary_queue.append(packet)
        return True

    def size(self):
        return sum(len(q) for q in [self.critical_queue, self.delay_sensitive_queue,
                                    self.reliability_sensitive_queue, self.ordinary_queue])

    def manage_timeouts(self, current_time, estimated_tx_time):
        lower_queues = [self.delay_sensitive_queue, self.reliability_sensitive_queue, self.ordinary_queue]
        for q in lower_queues:
            for i in range(len(q) - 1, -1, -1):
                if (q[i].absolute_deadline - current_time) <= estimated_tx_time:
                    p = q.pop(i)
                    p.priority_tag = 'High+'
                    self.critical_queue.append(p)

    def get_next_packet_for_transmission(self):
        for q in [self.critical_queue, self.delay_sensitive_queue, self.reliability_sensitive_queue, self.ordinary_queue]:
            if q: return q.pop(0)
        return None

class Node:
    def __init__(self, node_id, initial_energy=100.0):
        self.node_id = node_id
        self.energy = initial_energy
        self.neighbor_list = {}
        self.network_env = None
        self.scheduler = PriorityScheduler()
        
        # Dynamic Trackers (Eq. 20, 23)
        self.total_tx = 0
        self.success_tx = 0
        self.link_reliability = 1.0 
        self.link_stats = {}
        self.pkt_in = 0
        self.pkt_out = 0
        self.avg_traffic_load = 0.0

    def broadcast_hello(self, current_time):
        packet = HelloPacket(self.node_id, self.node_id, current_time, self.energy)
        self.network_env.broadcast(self, packet)

    def receive_hello(self, packet):
        # 1-hop Discovery
        if packet.sn == packet.rn:
            self.neighbor_list[packet.sn] = {'hops': 1, 'energy': packet.energy, 'relay': packet.sn}
            # Rebroadcast for 2-hop discovery (Algorithm 1, line 7)
            relay_packet = HelloPacket(packet.sn, self.node_id, packet.t_a, self.energy)
            self.network_env.broadcast(self, relay_packet)
        # 2-hop Discovery
        elif packet.sn != self.node_id:
            if packet.sn not in self.neighbor_list:
                self.neighbor_list[packet.sn] = {'hops': 2, 'energy': packet.energy, 'relay': packet.rn}

    def get_dynamic_delay(self):
        """Eq. 15 & 16: Dynamic delay based on queue size."""
        pkt_b = self.scheduler.size()
        t_nd = 0.5 * ((pkt_b / QUEUE_CAPACITY) + 1) + 0.5 * 0.01
        delay_seconds = 1 - (1 / (t_nd + 1))
        # Scale down to a realistic ~10-20ms base delay
        return max(0.005, delay_seconds * 0.05)

    def get_link_reliability(self, neighbor_id):
        stats = self.link_stats.get(neighbor_id)
        if not stats or stats['total'] == 0:
            return 1.0
        return stats['success'] / stats['total']

    def consume_tx_energy(self, size_bytes=PACKET_SIZE_BYTES):
        before_energy = self.energy
        self.energy = max(0.0, self.energy - (TX_ENERGY_PER_BYTE_J * size_bytes))
        return before_energy - self.energy

    def consume_rx_energy(self, size_bytes=PACKET_SIZE_BYTES):
        before_energy = self.energy
        self.energy = max(0.0, self.energy - (RX_ENERGY_PER_BYTE_J * size_bytes))
        return before_energy - self.energy

    def record_transmission(self, success=True, neighbor_id=None):
        """Eq. 20 & 23: Update metrics after every hop."""
        self.total_tx += 1
        self.pkt_out += 1
        if success: self.success_tx += 1
        if neighbor_id is not None:
            stats = self.link_stats.setdefault(neighbor_id, {'total': 0, 'success': 0})
            stats['total'] += 1
            if success:
                stats['success'] += 1
        delta = 0.4 # Paper weighting factor
        self.link_reliability = (1 - delta) * self.link_reliability + delta * (self.success_tx / self.total_tx)
        if self.pkt_out > 0:
            self.avg_traffic_load = self.pkt_in / self.pkt_out
