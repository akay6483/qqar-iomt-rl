from algorithms.node import Node, DataPacket

def test_priority_scheduler():
    print("--- Testing Algorithm 2: Priority Scheduling ---\n")
    
    # 1. Initialize a single node
    node = Node(node_id=0)
    current_time = 0.0
    
    # 2. Create 4 packets (Low, Med, High, High+)
    # We enqueue them backwards (Lowest priority first) to see if the node reorders them.
    # We give the 'Med' packet a very short deadline to force a timeout promotion.
    p1 = DataPacket(packet_id=101, source_id=0, sink_id=199, priority_tag='Low', creation_time=0.0, deadline_duration=5.0)
    p2 = DataPacket(packet_id=102, source_id=0, sink_id=199, priority_tag='Med', creation_time=0.0, deadline_duration=0.1) # Dying packet!
    p3 = DataPacket(packet_id=103, source_id=0, sink_id=199, priority_tag='High', creation_time=0.0, deadline_duration=2.0)
    p4 = DataPacket(packet_id=104, source_id=0, sink_id=199, priority_tag='High+', creation_time=0.0, deadline_duration=1.0)
    
    print("Injecting packets into Node 0's buffer...")
    node.scheduler.enqueue_packet(p1)
    node.scheduler.enqueue_packet(p2)
    node.scheduler.enqueue_packet(p3)
    node.scheduler.enqueue_packet(p4)
    
    print(f"Queues loaded. Total packets: 4")
    
    # 3. Simulate a time tick where the 'Med' packet is about to expire
    current_time = 0.05
    estimated_tx_time = 0.1
    print(f"\nTime advances to {current_time}s. Running timeout manager...")
    
    # Algorithm 2 timeout management
    node.scheduler.manage_timeouts(current_time, estimated_tx_time)
    
    # 4. Pop the packets out of the node to transmit
    print("\nExtracting packets for transmission (Algorithm 3 flow):")
    for _ in range(4):
        pkt = node.scheduler.get_next_packet_for_transmission()
        if pkt:
            print(f"Transmitting Packet {pkt.packet_id} | Final Priority: {pkt.priority_tag}")

if __name__ == "__main__":
    test_priority_scheduler()
