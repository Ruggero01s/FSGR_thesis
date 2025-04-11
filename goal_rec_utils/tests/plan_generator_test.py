import unittest
from unittest.mock import Mock, patch, PropertyMock
from goal_rec_utils.plan_generator import get_seed, get_actions, get_goal, fill_action_sequence, np
from goal_rec_utils.plan import Action


class TestGetSeed(unittest.TestCase):

    name = '../datset/logistics/tasks_simil_pereira/plans/p000001.pddl'

    def test_correct(self):
        self.assertEqual(get_seed('c'), 99)
        self.assertEqual(get_seed('et'), ord('t')+ord('e'))
        seed = get_seed(self.name)
        self.assertIsInstance(seed, int)
        self.assertEqual(seed, get_seed(self.name))


class TestGetActions(unittest.TestCase):
    np.random.seed(4)
    actions = list()
    for i in range(10):
        a = Mock()
        a.name = f'a{i}'
        actions.append(a)
    actions_dict = {k.name: i+1 for i, k in enumerate(actions)}

    def test_correct(self):
        self.assertEqual(len(get_actions(self.actions, 0.5, self.actions_dict)), 5)
        np.random.seed(4)
        p = np.random.uniform(0.3, 0.7)
        a = get_actions(self.actions, p, self.actions_dict)
        np.random.seed(4)
        p = np.random.uniform(0.3, 0.7)
        a1 = get_actions(self.actions, p, self.actions_dict)
        self.assertEqual(a1, a)

    def test_borders(self):
        self.assertEqual(len(get_actions([self.actions[0]], 0.5, self.actions_dict)), 1)
        self.assertEqual(get_actions(self.actions, 10, self.actions_dict), [self.actions_dict[a.name] for a in self.actions])
        self.assertEqual(len(get_actions(self.actions, -3, self.actions_dict)), 1)
        self.assertEqual(get_actions([], 1, self.actions_dict), [])
        self.assertEqual(get_actions(None, 0.9, self.actions_dict), [])

class TestFillActionSequence(unittest.TestCase):

    action_seq = range(1, 11)

    def reset_x(self, shape:tuple):
        self.X = np.zeros(shape)

    def testFillActionSequence(self):
        self.reset_x((1,10))
        fill_action_sequence(self.X, 10, self.action_seq, 0)
        self.assertTrue(np.all(self.X == np.asanyarray([range(1,11)])))
        self.reset_x((1,8))
        fill_action_sequence(self.X, 8, self.action_seq, 0)
        self.assertTrue(np.all(self.X == np.asanyarray([range(1, 9)])))
        self.reset_x((1,8))
        fill_action_sequence(self.X, 8, [], 0)
        self.assertTrue(np.all(self.X == np.zeros((1,8))))



