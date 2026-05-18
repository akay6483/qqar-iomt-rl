import unittest

from algorithms.network import NetworkEnv
from algorithms.node import DataPacket, Node
from algorithms.paper_config import HELLO_PACKET_BYTES
from experiment.simulation_engine import SimulationEngine


class FirstCandidateAgent:
    def select_best_route(self, current_id, candidates, sinks):
        return candidates[0] if candidates else None


def build_two_node_env():
    env = NetworkEnv(area_size=10, num_nodes=2, tx_range=10)
    for node_id, pos in [(0, (0, 0)), (1, (1, 0))]:
        node = Node(node_id=node_id)
        node.network_env = env
        env.nodes[node_id] = node
        env.graph.add_node(node_id, pos=pos)
    env.graph.add_edge(0, 1, distance=1.0)
    env.nodes[0].neighbor_list[1] = {'hops': 1, 'energy': 100.0, 'relay': 1}
    env.nodes[1].neighbor_list[0] = {'hops': 1, 'energy': 100.0, 'relay': 0}
    return env


class SimulationAccountingTests(unittest.TestCase):
    def test_hello_discovery_counts_overhead_and_energy(self):
        env = build_two_node_env()

        env.nodes[0].broadcast_hello(1.0)

        self.assertEqual(env.routing_overhead_bytes, HELLO_PACKET_BYTES * 2)
        self.assertGreater(env.control_energy_consumed, 0.0)
        self.assertLess(env.nodes[0].energy, 100.0)
        self.assertLess(env.nodes[1].energy, 100.0)

    def test_low_battery_runtime_energy_uses_actual_delta(self):
        env = build_two_node_env()
        engine = SimulationEngine(
            env,
            FirstCandidateAgent(),
            sinks=[1],
            wban_nodes=[0],
            data_rate=0,
            max_time=0,
            initial_energy=0.1,
        )

        success, _ = engine._transmit_hop(0, 1)

        self.assertTrue(success)
        self.assertAlmostEqual(engine.energy_consumed, 0.2)
        self.assertEqual(env.nodes[0].energy, 0.0)
        self.assertEqual(env.nodes[1].energy, 0.0)

    def test_initial_energy_mapping_preserves_training_depletion(self):
        env = build_two_node_env()
        SimulationEngine(
            env,
            FirstCandidateAgent(),
            sinks=[1],
            wban_nodes=[0],
            data_rate=0,
            max_time=0,
            initial_energy={0: 12.5, 1: 40.0},
        )

        self.assertEqual(env.nodes[0].energy, 12.5)
        self.assertEqual(env.nodes[1].energy, 40.0)

    def test_expired_after_hop_delay_is_not_delivered(self):
        env = build_two_node_env()
        engine = SimulationEngine(
            env,
            FirstCandidateAgent(),
            sinks=[1],
            wban_nodes=[0],
            data_rate=0,
            max_time=0.01,
            channel_seed=1,
        )
        pkt = DataPacket(1, 0, 1, 'High+', creation_time=0.0, deadline_duration=0.001)
        env.nodes[0].scheduler.enqueue_packet(pkt)
        engine.active_packets[pkt.packet_id] = {'creation': 0.0, 'hops': 0}
        engine.generated_packets = 1

        pdr, _, _, _ = engine.run()

        self.assertEqual(pdr, 0.0)
        self.assertEqual(engine.successful_packets, 0)
        self.assertEqual(engine.expired_packets, 1)
        self.assertEqual(engine.active_packets, {})


if __name__ == '__main__':
    unittest.main()
