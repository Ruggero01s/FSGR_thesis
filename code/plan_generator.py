from keras.utils import Sequence
import numpy as np
import random
import copy

class PlanGenerator_old(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim), len(self.dizionario)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            actions = get_actions(plan.actions, self.perc, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.perc = perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


class PlanGenerator(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            actions = get_actions(plan.actions, self.perc, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.perc = perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


class PlanGeneratorMultiPerc(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            seed = plan.plan_name.rsplit('p',1)[1]
            seed = seed.split('_', 1)[0]
            seed = seed.rsplit('.', 1)[0]
            np.random.seed(int(seed))
            p = np.random.uniform(self.min_perc, self.perc)
            actions = get_actions(plan.actions, p, self.dizionario)
            print(f"plan actions: {actions}")
            fill_action_sequence(X, self.max_dim, actions, i)
            #todo fill action sequence does something?
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
            #todo check this output
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, min_perc, max_perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.perc = max_perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)

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
    # return [dizionario[a.name] for a in np.take(actions, indexes)] #depending on action.py class implementation of pickled plans
    # If actions are strings use the following line:
    #todo check this return
    return [dizionario[a.upper()] for a in np.take(actions, indexes)]


def fill_action_sequence(X, max_dim, actions, i):
    #todo check this function
    for j in range(max_dim):
        if j < len(actions):
            X[i][j] = actions[j]
        else:
            if type(actions[0]) == int:
                X[i][j] = 0
            else:
                X[i][j] = np.zeros(shape=(len(actions[0]),))

def get_goal(g, dizionario_goal):
    #todo check this function
    goal = np.zeros(len(dizionario_goal))
    for subgoal in g:
        if subgoal in dizionario_goal:
            goal = goal + dizionario_goal[subgoal]
    return goal

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


# to generate more
class PlanGeneratorMultiPercAugmented(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            if plan.is_complete:
                actions = get_actions(plan.actions, 1, self.dizionario)
            else:
                seed = get_seed(plan.plan_name)
                np.random.seed(int(seed))
                p = np.random.uniform(self.min_perc, self.perc)
                actions = get_actions(plan.actions, p, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def augment_plans(self, plans, num_plans):
        augmented_plans = []
        for p in plans:
            for  i in range(num_plans):
                plan = copy.deepcopy(p)
                plan.is_complete = False
                plan.plan_name = plan.plan_name + str(i)
                augmented_plans.append(plan)
            if self.add_complete:
                plan = copy.deepcopy(p)
                plan.is_complete = True
                augmented_plans.append(plan)
        return augmented_plans


    def __init__(self, plans, dizionario, dizionario_goal, num_plans, batch_size, max_dim, min_perc, max_perc, add_complete=True, shuffle=True):
        
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.num_plans = num_plans
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.perc = max_perc
        self.add_complete = add_complete
        self.shuffle = shuffle
        self.plans = self.augment_plans(plans, num_plans)

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


np.random.seed(43)
random.seed(43)
