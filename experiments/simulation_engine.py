import random
from algorithms.node import DataPacket, PriorityScheduler

class SimulationEngine:
    def __init__(self, env, agent, sinks, wban_nodes, data_rate, max_time=10.0):
        self.env = env
        self.agent = agent
        self.sinks = sinks
        self.wban_nodes = wban_nodes
        self.data_rate = data_rate
        self.max_time = max_time
        self.current_time = 0.0
        self.time_step = 0.01
        
        for node in self.env.nodes.values():
            node.scheduler = PriorityScheduler()
            node.pkt_in = 0
            node.pkt_out = 0
            node.total_tx = 0
            node.success_tx = 0
            node.link_reliability = 1.0
            node.avg_traffic_load = 0.0
            node.energy = 100.0 # Reset energy for the new simulation run

        self.successful_packets = 0
        self.generated_packets = 0
        self.total_delay = 0.0
        self.total_hops = 0
        self.energy_consumed = 0.0
        self.active_packets = {}

    def generate_traffic(self):
        # Calculate exactly how many packets should spawn this exact millisecond
        expected_packets = (self.data_rate * len(self.wban_nodes)) * self.time_step
        num_packets = int(expected_packets) # Guaranteed packets
        
        # Fractional chance to spawn one extra packet
        if random.random() < (expected_packets - num_packets):
            num_packets += 1
            
        # Spawn the correct volume of traffic!
        for _ in range(num_packets):
            self.generated_packets += 1
            src = random.choice(self.wban_nodes)
            snk = random.choice(self.sinks)
            
            rv = random.random()
            if rv < 0.05: tag, dl = 'High+', 1.0
            elif rv < 0.15: tag, dl = 'High', 2.0
            elif rv < 0.40: tag, dl = 'Med', 5.0
            else: tag, dl = 'Low', 10.0
                
            pkt = DataPacket(self.generated_packets, src, snk, tag, self.current_time, dl)
            self.active_packets[pkt.packet_id] = {'creation': self.current_time, 'hops': 0}
            self.env.nodes[src].scheduler.enqueue_packet(pkt)

    def run(self):
        while self.current_time < self.max_time:
            self.generate_traffic()
            for node_id, node in self.env.nodes.items():
                if node_id in self.sinks: continue
                
                node.scheduler.manage_timeouts(self.current_time, self.time_step)
                pkt = node.scheduler.get_next_packet_for_transmission()
                
                if pkt:
                    if self.current_time >= pkt.absolute_deadline: continue
                    
                    cur_dist = min([self.env.get_distance(node_id, s) for s in self.sinks])
                    candidates = [n for n in node.neighbor_list.keys() if cur_dist > min([self.env.get_distance(n, s) for s in self.sinks])]
                    
                    if not candidates: continue
                    
                    nxt = self.agent.select_best_route(node_id, candidates, self.sinks)
                    if not nxt: continue # Failsafe for PlainQ
                    
                    self.active_packets[pkt.packet_id]['hops'] += 1
                    
                    # Energy Tracking
                    node.energy -= 1.055 # Tx/Rx energy 
                    self.energy_consumed += 1.055 
                    
                    # Simulate MAC layer collisions. Heavier load = higher drop chance
                    collision_chance = min(0.6, node.avg_traffic_load * 0.15)
                    
                    if random.random() > collision_chance:
                        node.record_transmission(success=True)
                        self.env.nodes[nxt].pkt_in += 1
                        
                        if nxt in self.sinks:
                            self.successful_packets += 1
                            self.total_hops += self.active_packets[pkt.packet_id]['hops']
                            self.total_delay += (self.current_time - pkt.creation_time) * 1000
                        else:
                            self.env.nodes[nxt].scheduler.enqueue_packet(pkt)
                    else:
                        node.record_transmission(success=False)
                        
            self.current_time += self.time_step
            
        return self._metrics()

    def _metrics(self):
        pdr = (self.successful_packets / max(1, self.generated_packets)) * 100
        return pdr, self.total_delay/max(1,self.successful_packets), self.total_hops/max(1,self.successful_packets), self.energy_consumed