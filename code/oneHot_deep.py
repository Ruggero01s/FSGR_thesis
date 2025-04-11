import numpy as np
import random


def add_action_dictionary(action, dictionary):
    if not action in dictionary:
        dictionary[action] = len(dictionary) + 1


def oneHot(action, dictionary):
    encode = np.zeros(len(dictionary))
    encode[dictionary[action]] = 1
    return encode


def safe_number(action, dictionary):
    return dictionary[action]


def create_dictionary(plans):
    dictionary = {}
    for p in plans:
        for action in p.actions:
            add_action_dictionary(action.name, dictionary)
    return dictionary


def add_goal_dictionary(g, dictionary):
    goal = ""
    for subgoal in g:
        goal = goal + subgoal
    if goal not in dictionary:
        dictionary[goal] = len(dictionary)


def add_goal_dictionary2(g, dictionary):
    g.sort()
    goal = ""
    for subgoal in g:
        goal = goal + subgoal
    if goal not in dictionary:
        dictionary[goal] = len(dictionary)


def create_dictionary_goals(goal):
    dictionary = {}
    for g in goal:
        add_goal_dictionary(g, dictionary)
    return dictionary


def create_dictionary_goals2(goal):
    dictionary = {}
    for g in goal:
        add_goal_dictionary2(g, dictionary)
    return dictionary


def shuffle_dictionary(dictionary):
    return dict(
        zip(
            list(dictionary.keys()),
            random.sample(list(dictionary.values()), len(dictionary)),
        )
    )


def oneHot_plans(plans, dictionary):
    for p in plans:
        for action in p.actions:
            action.code_action(safe_number(action.name, dictionary))


def completa_dizionario(dictionary):
    dim = len(dictionary)
    for key, value in dictionary.items():
        one_hot = np.zeros(dim)
        one_hot[value - 1] = 1
        dictionary[key] = one_hot
