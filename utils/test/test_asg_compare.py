import unittest
import rustworkx as rx

from utils.asg_compare import compare_asgs


class TestASGCompare(unittest.TestCase):
    def setUp(self) -> None:
        self.psg1 = rx.PyDAG()
        psg_ego = self.psg1.add_node('ego')
        self.psg1.add_child(psg_ego, 'car', 'near')
        self.psg1.add_child(psg_ego, 'car', 'near')
        self.rsv1 = rx.PyDAG()
        rsv_ego = self.rsv1.add_node('ego')
        self.rsv1.add_child(rsv_ego, 'car', 'left')
        self.rsv1.add_child(rsv_ego, 'SUV', 'right')
    
    def test_asg_equals(self):
        self.assertFalse(compare_asgs(self.psg1, self.rsv1))
        self.assertTrue(compare_asgs(self.psg1, self.psg1))


if __name__ == '__main__':
    unittest.main()
    