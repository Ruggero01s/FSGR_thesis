from tensorflow.keras.utils import Sequence
import numpy as np


class PlanGeneratorMultiPerc(Sequence):
    '''
    Tensorflow generator class.
    Generates tensors that represent a set of plans.
    '''

    def __getitem__(self, index):
        '''
        Get the i-th batch of plan

        Args:
            index: int representing the index of the batch to return

        Returns:
            A tuple (X, Y) where X is a tensor containing a batch of plans and Y is a tensor containing a batch
            of goals.
        '''
        batches = self.plans[index * self.batch_size : (index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            seed = get_seed(plan.plan_name)
            np.random.seed(int(seed))
            p = np.random.uniform(self.min_perc, self.perc)
            actions = get_actions(plan.actions, p, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        '''
        Computes the number of batches

        Returns:
            an int containing the number of batches
        '''
        return len(self.plans) // self.batch_size

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        min_perc,
        max_perc,
        shuffle=True,
    ):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.perc = max_perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''
        Updates indexes after an epoch
        '''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


def get_seed(string: str):
    '''
    Turns a string into an int that can be used as a seed

    Args:
        string: the string to transform in seed

    Returns:
        an integer containing the seed
    '''
    seed = 0
    for c in string:
        seed += ord(c)
    return seed


def get_actions(actions: list, perc: float, dizionario: dict):
    '''
    Get a sub-sequence made by a given percentage of elements in action.

    Args:
        actions: a list that contains the actions as strings
        perc: a float that represents the percentage of actions to keep. It must be from 0 to 1 included.
        dizionario: a dictionary that contains all the action labels and their corresponding unique indexes

    Returns:
        A list that contains the indexes of the actions in the sub-sequence

    '''
    if actions is None or len(actions) == 0:
        return []
    if perc > 1:
        perc = 1
    elif perc < 0:
        perc = 0
    size = int(np.ceil(len(actions) * perc))
    if size == 0:
        size = 1
    indexes = np.ones(size, dtype=int) * -1
    i = 0
    ind_list = list(range(len(actions)))
    np.random.shuffle(ind_list)
    while i < size:
        ind = ind_list.pop(0)
        if ind not in indexes:
            indexes[i] = ind
            i += 1
    indexes = np.sort(indexes)
    return [dizionario[a.name] for a in np.take(actions, indexes)]


def fill_action_sequence(X: np.ndarray, max_dim: int, actions: list, i: int):
    '''
    Fill the i-th row of tensor X with the indexes of the actions. The list is padded with zeros if its length is less
    than max_dim
    Args:
        X: ndarray that contains the tensor of actions indexes
        max_dim: an integer that represents the maximum dimension of a list of actions
        actions: a list of action indexes
        i: an int that represents the row number where to insert the action list
    '''
    for j in range(max_dim):
        if j < len(actions):
            X[i][j] = actions[j]
        else:
            if len(actions) == 0:
                X[i] = np.zeros((max_dim,))
            elif type(actions[0]) == int:
                X[i][j] = 0
            else:
                X[i][j] = np.zeros(shape=(len(actions[0]),))


def get_goal(g: list, dizionario_goal: dict):
    '''
    Transforms a goal into a binary list
    Args:
        g: a list of facts in the goal state represented as strings
        dizionario_goal: a dictionary that contains a unique one-hot vector for each fact

    Returns:
        A binary vector that represents the goal
    '''
    goal = np.zeros(len(dizionario_goal))
    for subgoal in g:
        goal = goal + dizionario_goal[subgoal]
    return goal


np.random.seed(43)