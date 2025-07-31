import os

from os.path import  join, isdir
from plan import Plan
from action import Action
from utils import load_from_folder
from multiprocess import Pool
import random
from logging import exception
import re
import shutil
import multiprocess
random.seed(42)

save_dir = './generated_gr_dataset/'
data_base_dir = '../datasets/'
domain = 'logistics'
results_dir = f"{save_dir}/{domain}/"   
source_dir = f"{join(data_base_dir, domain)}/plans/" 
print('Domain dir:', source_dir)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

plans_to_process = 50000 # number of plans to process
versions_per_plan = 5 # number of versions per each plan
number_of_goals = 4 # number of goals per each new plan
test = False # test will process only 3 plans
#rec_classes = [[0,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.999]] # classes of recognizability
rec_classes = [[0,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8], [0.8, 1]] # classes of recognizability

#to clear the results directory
for folder in os.listdir(results_dir):
    #remove directory if it exists along with its content
    if isdir(join(results_dir, folder)):
        shutil.rmtree(join(results_dir, folder))
        
        
class PDDLProblem:
    """
    Represents a PDDL problem.

    Attributes:
        name (str): the name of the problem
        domain (str): the domain name
        objects (dict): mapping from types to lists of object names
        init (list): list of predicate tuples representing the initial state
        goal (list): list of predicate tuples representing the goal state
    """
    def __init__(self, name, domain, objects, init, goal):
        self.name = name
        self.domain = domain
        self.objects = objects
        self.init = init
        self.goal = goal

    def __repr__(self):
        return (f"PDDLProblem(name={self.name!r}, domain={self.domain!r}, "
                f"objects={self.objects!r}, init={self.init!r}, goal={self.goal!r})")


def tokenize(pddl_str):
    """
    Converts PDDL string into a list of tokens.
    Strips comments and splits parentheses as separate tokens.
    """
    # Remove comments
    pddl_str = re.sub(r";.*", "", pddl_str)
    # Tokenize parentheses and symbols
    tokens = re.findall(r"\(|\)|[^\s()]+", pddl_str)
    return tokens


def parse(tokens):
    """
    Parses a list of tokens into a nested list (S-expression).
    """
    def parse_expr(index):
        assert tokens[index] == '(', f"Expected '(', got {tokens[index]}"
        index += 1
        expr = []
        while tokens[index] != ')':
            if tokens[index] == '(':
                subexpr, index = parse_expr(index)
                expr.append(subexpr)
            else:
                expr.append(tokens[index])
                index += 1
        return expr, index + 1

    sexpr, next_index = parse_expr(0)
    return sexpr


def _extract_typed_objects(tokens):
    """
    Helper to process typed object lists: [obj1, obj2, '-', type, ...]
    Returns dict mapping types to object lists.
    """
    objs = {}
    current = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == '-':
            obj_type = tokens[i+1]
            objs.setdefault(obj_type, []).extend(current)
            current = []
            i += 2
        else:
            current.append(tok)
            i += 1
    # Untyped objects (no dash): assign to None
    if current:
        objs.setdefault(None, []).extend(current)
    return objs


def parse_pddl_problem(file_path):
    """
    Parse a PDDL problem file and return a PDDLProblem instance.
    """
    with open(file_path) as f:
        data = f.read()

    tokens = tokenize(data)
    sexpr = parse(tokens)

    # Skip 'define', first element
    assert sexpr[0] == 'define', "Not a PDDL define file"

    # Initialize placeholders
    problem_name = None
    domain_name = None
    objects = {}
    init = []
    goal = []

    # Iterate over top-level sections
    for section in sexpr[1:]:
        if isinstance(section, list) and section:
            key = section[0]
            if key == 'problem':  # (problem name)
                # section could be ['problem', 'name'] or ['problem', 'name', ...]
                problem_name = section[1]

            elif key == ':domain':
                domain_name = section[1]

            elif key == ':objects':
                # section[1:] is flat list of names, '-', types
                objects = _extract_typed_objects(section[1:])

            elif key == ':init':
                # section[1:] are lists of atoms
                for atom in section[1:]:
                    if isinstance(atom, list) and atom:
                        pred = atom[0]
                        args = atom[1:]
                        fluent = pred
                        for arg in args:
                            fluent += f" {arg}"
                        init.append(fluent)
                        

            elif key == ':goal':
                # section[1] may be an 'and' list or a single atom
                goal_expr = section[1]
                if isinstance(goal_expr, list) and goal_expr[0] == 'and':
                    atoms = goal_expr[1:]
                else:
                    atoms = [goal_expr]
                for atom in atoms:
                    if isinstance(atom, list) and atom:
                        pred = atom[0]
                        args = atom[1:]
                        fluent = pred
                        for arg in args:
                            fluent += f" {arg}"
                        goal.append(fluent)

    name = re.search(r"(p\d+)(?=\.)", file_path).group(1)
    # Create PDDLProblem instance
    return PDDLProblem(name, domain_name, objects, init, goal)


def compute_recognizability(current_goal_state, goal_state_list):
    """
    Compute the difficulty of a plan.
    :param current_goal_state: The goal state for which we calculate the recognizability.
    :param goal_state_list: The list of goal states to use for the computation.
    :return: The recognizability of the plan.
    """
    
    #the current goal state must be in the goal state list
    if current_goal_state not in goal_state_list:
        raise ValueError(f"current_goal_state {current_goal_state} must be included in goal_state_list {goal_state_list}")
    
    #min and max to use for normalization
    #teoretical min recognizability is when all the fluent in current are present in each goal state in the goal state list
    #the formula becomes 1/n+1/n+1/n+...+1/n = len(current_goal_state) / len(goal_state_list)
    #teoretical max recognizability is when all the fluent in current are not present anywhere in the goal state list
    #formula is 1/1 + 1/1 + 1/1 ... + 1/1 = 1 * len(current_goal_state)
    
    #? min max accettabili ovvero
    #?  per il min almeno uno deve essere diverso, non vogliamo rec=0 , ok) 
    #?  per il max invece?   
    #?       è accettabile avere tutti i fluenti diversi quindi rec=1, o deve essercene uno in uno dei goal delle versioni che è uguale?
    
    min_recognizability = 1/(len(goal_state_list)) * (len(current_goal_state)) # tutti i fluenti uguali
    #min è quando sono tutti diversi di almeno uno
    #min_recognizability  = 1/(len(goal_state_list)) * (len(current_goal_state)-1) + 1/(len(goal_state_list)-1) # tutti fluenti presenti in tutti i goal a parte uno che non è presente da qualche parte
    #? (len(current_goal_state)-2) * (len(goal_state_list)+1)/len(goal_state_list) # questa è invece la tua formula per la massima uniqueness
    #? perchè la massima uniqueness non è quando i fluenti del base non compaiono da nessun'altra parte? o se quel caso non va bene allora quello in cui ce n'è solo uno uguale?
    max_recognizability  = 1*len(current_goal_state) # tutti fluenti diversi
    #max_recognizability  = 1*(len(current_goal_state)-1) + 1/2 # tutti fluenti diversi a parte uno che ha un doppione da qualche parte
    
    #todo sistemare formula hardcodando max e min
    #k=2, n=6 = 7/12 
    #k=3, n=6 = 3/4
    #k=4, n=6 = #da runnare
    #todo
    
    # print(f"min_recognizability: {min_recognizability}")
    # print(f"max_recognizability: {max_recognizability}")
    
    sum = 0
    
    # too see the sum of fractions
    # debug_string = ""
    #need to count how many times the current goal fluent is in the goal state list
    for current_goal_fluent in current_goal_state:
        count = 0
        for goal_state in goal_state_list:
            for goal_fluent in goal_state:       
                if current_goal_fluent==goal_fluent:
                    count += 1
                    break
        sum += 1/count
    #     debug_string += f" + 1/{count}"
    
    # debug_string += f" = {sum}"
    # debug_string = debug_string[3:]
    #print(f"Unscaled recognizability: {sum}")
    
    #normalize the recognizability #? should be ok 
    recognizability = (sum-min_recognizability) / (max_recognizability-min_recognizability)
    # print(debug_string)
    return round(recognizability, 4)


def check_if_goal_state_in_init(init, goal_state):
    goal_present_in_init = True
    for goal_fluent in goal_state:
        # print(base_goal_fluent)
        # print(problem.og_plan.initial_state)
        if goal_fluent in init:
            continue
        else:
            goal_present_in_init = False
            break
    
    return goal_present_in_init

def create_goal_state_list(initial_state, package_for_goal_set, pos_for_goal_set, number_of_goals, versions_per_plan):
    """Create a list of goal states.
    :param package_for_goal_set: The set of packages to use for the goal state.
    :param pos_for_goal_set: The set of positions to use for the goal state.
    :param number_of_goals: The number of goals to generate.
    :param versions_per_plan: The number of versions to generate.
    :return: A list of goal states.
    """	
    goal_state_list = []
    for i in range(0, versions_per_plan + 1):
        # generate a random goal state
        goal_state = generate_goal_state(initial_state, package_for_goal_set, pos_for_goal_set, number_of_goals)
            
        goal_state_list.append(goal_state)
        
    return goal_state_list
        
def generate_goal_state(initial_state, package_for_goal_set, pos_for_goal_set, number_of_goals):
    """Generate a random goal state.	
    :param package_for_goal_set: The set of packages to use for the goal state.
    :param pos_for_goal_set: The set of positions to use for the goal state.
    :param number_of_goals: The number of fluents to generate for the goal state.
    :return: A set of fluents representing a goal state.
    """
    while True:
        goal_state = set()
        package_for_goal_set_copy = package_for_goal_set.copy()
        pos_for_goal_set_copy = pos_for_goal_set.copy()
        for _ in range(number_of_goals):
            random_package = random.choice(list(package_for_goal_set_copy))
            package_for_goal_set_copy.remove(random_package)
            random_pos = random.choice(list(pos_for_goal_set_copy))
            
            #? can objects be in same position? assuming yes
            #? pos_for_goal_set_copy.remove(random_pos) 
            goal_state.add(f"at {random_package} {random_pos}") 
        
        if not check_if_goal_state_in_init(initial_state, goal_state):
            break
    return list(goal_state) 

#to calculate how precise the generation is
global_counter = 0

running_sum_rec_error = 0


def check_consistency(goal_state, prefix="obj"):
    """	
    Check if the goal state is consistent by checking if it has same obj used more than once.
    :param goal_state: The goal state to check.
    :return: True if the goal state is consistent, False otherwise.
    """
    objects = []
    for fluent in goal_state:
        obj = re.search(rf"{prefix}\d+", fluent)
        objects.append(obj.group(0))
    return len(objects) == len(set(objects))
        

def check_if_fluent_is_usable(fluent_to_add, goal_state):
    """
    Check if the fluent to add is usable in the goal state.
    This means checking if fluent is not already in the goal state and if the object in fluent is not already in the goal state.
    :param fluent_to_add: The fluent to add.
    :param goal_state: The goal state to check.
    :return: True if the fluent to add is usable, False otherwise.
    """
    #check if goal state is a list of strings
    if isinstance(fluent_to_add, list):
        raise ValueError(f"fluent to add is a list: {fluent_to_add}")
    for fluent in goal_state:
        if fluent == fluent_to_add:
            return False
        elif check_same_object_in_fluents(fluent_to_add, fluent, prefix="obj"):
            return False
    return True

def check_same_object_in_fluents(fluent1, fluent2, prefix="obj"):
    """
    Check if the object in fluent1 is in fluent2.
    :return: True if the object in fluent1 is in fluent2, False otherwise.
    """
    #extract the object from the fluent using regex
    obj1 = re.search(rf"{prefix}\d+", fluent1).group(0)
    obj2 = re.search(rf"{prefix}\d+", fluent2).group(0)

    if obj1 == obj2:
        return True
    return False

def check_duplicates(goal_state_list):
    """
    Check if the goal state list has duplicates.
    :param goal_state_list: The goal state list to check.
    :return: True if the goal state list has duplicates, False otherwise.
    """
    #check if goal state is a list of strings
    if isinstance(goal_state_list, str):
        raise ValueError(f"goal state list is a string: {goal_state_list}")
    
    #check if there are duplicates in the goal state list
    return len(goal_state_list) != len(set(tuple(sorted(g)) for g in goal_state_list))

def expand_candidates_list(goal_state, base_goal_state, package_for_goal_set, pos_for_goal_set):
    """
    Expand candidates list by generating all possible fluent
    """
    #generate all possible fluents
    all_possible_fluents = []
    for package in package_for_goal_set:
        for pos in pos_for_goal_set:
            all_possible_fluents.append(f"at {package} {pos}")
    
    #remove the fluents that are already in the goal state
    for fluent in goal_state:
        if fluent in all_possible_fluents:
            all_possible_fluents.remove(fluent)
    
    #remove the fluents that are already in the base goal state
    for fluent in base_goal_state:
        if fluent in all_possible_fluents:
            all_possible_fluents.remove(fluent)
            
    for fluent in all_possible_fluents:
        if not check_if_fluent_is_usable(fluent, goal_state):
            all_possible_fluents.remove(fluent)
    
    return all_possible_fluents

def adapt_goal_state_list_to_recognizability(initial_state, base_goal_state, goal_state_list, 
                                             package_for_goal_set, pos_for_goal_set, 
                                             number_of_goals, rec_target=[0, 1], 
                                             randomness_patience=5, regeneration_patience=3):
    """
    Adapt the goal state list to the recognizability. 
    This is done by swapping fluents between the base goal state and the goal state list.
    If we have to reduce recognisability, we swap a random fluent in the goal state list with a fluent from the base goal state.
    If we have to increase recognisability, we swap a fluent that is in the base goal state and in also in a goal in goal state list with a random possible fluent.
    This does not guarantee that the recognizability will be in the target range, but it will be close.
    It can happen that a goal state will be stuck in a local minimum, so we regenerate it to try staring from another point.
    We keep track of which regenerations we have done and if we reach the patience limit, we will use the one that is closest to the target recognizability.
    :param base_goal_state: The base goal state.
    :param goal_state_list: The list of goal states to use to compute recognizability, without base goal state.
    :param randomness_patience: The number of times we can try to adapt the goal state list before regenerating it. This will usually exhaust if we have many states that are stuck in a local minimum. If we reach this limit, we will regenerate the next stuck goal state.
    :param regeneration_patience: The number of times we can regenerate a goal state before giving up.
    :param package_for_goal_set: The set of packages to use for the goal state.
    :param pos_for_goal_set: The set of positions to use for the goal state.
    :param number_of_goals: The number of goals to generate.
    :param rec_target: The target range of recognizability.
    :return: The adapted goal state list.
    """
    
    if rec_target[0] > rec_target[1]:
        raise ValueError(f"rec_target[0] {rec_target[0]} must be less than rec_target[1] {rec_target[1]}")
    
    #variables to keep track of errors
    global global_counter
    global running_sum_rec_error
    
    
    randomness_patience_constant = randomness_patience
    
    #to keep track of the goal states that we have regenerated
    goal_regeneration_dict = {}
    
    #we identify all the fluents that are in the goal state list    
    all_goal_fluents = []
    for goal_state in goal_state_list:
        for fluent in goal_state:
            if fluent not in all_goal_fluents:
                all_goal_fluents.append(fluent)
    
    #we identify all the fluents that are not in the base goal state but are in the goal state list
    non_base_goal_fluents = []
    for fluent in all_goal_fluents:
        if fluent not in base_goal_state:
            non_base_goal_fluents.append(fluent)
    
    #starting recognizability
    running_recognizability = compute_recognizability(base_goal_state, [base_goal_state] + goal_state_list)
    #print(f"Starting recognizability: {running_recognizability}, range is {recognizability}")
    
    #while the recognizability is not in the target range, we will keep adapting the goal state list
    while running_recognizability < rec_target[0] or running_recognizability > rec_target[1]:
        #print(f"Running recognizability start of step: {running_recognizability}, range is {rec_target}")
        
        #choose a random goal state from the list
        goal_state = random.choice(goal_state_list)
        
        if running_recognizability > rec_target[1]:
            #rec too high: find a goal state in goal_state_list that has a fluent that is not in the base goal state
            #swap it with one from base_goal_fluents                
            
            #builds list of fluents that are in the base goal state but not in this goal state and don't introduce inconsistencies
            usable_base_goal_fluents = []
            for fluent in base_goal_state:  
                if check_if_fluent_is_usable(fluent, goal_state):
                    usable_base_goal_fluents.append(fluent)
                            
            if len(usable_base_goal_fluents) > 0:                
                
                #builds list of fluents that are in the goal state but not in the base goal state
                candidates_list = []
                for fluent in goal_state:
                    if fluent not in base_goal_state:
                        candidates_list.append(fluent)
                
                fluent_from_base_goal = random.choice(usable_base_goal_fluents)
                
                store_candidates_list = candidates_list.copy()
                
                #if there is at least one fluent to swap   
                if len(candidates_list) > 0:
                    
                    #we will do a while sequence to find a swap that does not introduce inconsistencies, creates duplicates or is already satisfied in the initial state
                    while True:
                        
                        # debug_goal_state = goal_state.copy() #debug
                        
                        if len(candidates_list) > 0:
                            random_fluent = random.choice(candidates_list)
                        else:
                            break
                        
                        # before = check_consistency(goal_state) #debug
                        
                        #we create what the state list will look like after the swap
                        proposed_goal_state = goal_state.copy()
                        proposed_goal_state_list = goal_state_list.copy()
                        proposed_goal_state_list[proposed_goal_state_list.index(goal_state)] = proposed_goal_state
                        
                        proposed_goal_state[proposed_goal_state.index(random_fluent)] = fluent_from_base_goal
                        
                        #then check if it's ok
                        if not check_duplicates(proposed_goal_state_list) and not check_if_goal_state_in_init(initial_state, proposed_goal_state):
                            break
                        else:
                            #if it's not we cannot use this candidate with this fluent
                            candidates_list.remove(random_fluent)
                            if len(candidates_list) == 0:
                                #if we finish the candidates we reset the list and try another usable fluent
                                candidates_list = store_candidates_list.copy()
                                usable_base_goal_fluents.remove(fluent_from_base_goal)
                                if len(usable_base_goal_fluents) == 0:
                                    #print(f"Candidate list is empty because of duplicates, no fluent from base goal is applicable breaking") # debug
                                    break
                                else:
                                    fluent_from_base_goal = random.choice(usable_base_goal_fluents)
                    
                    #if all is fine we make the swap in the list                             
                    goal_state_list[goal_state_list.index(goal_state)] = proposed_goal_state 
                    # after = check_consistency(goal_state) #debug
                    # if before == True and after == False: #debug
                    #     print(f"|>|Goal state is not consistent: old goal state{debug_goal_state}\n\t base_goal_state: {base_goal_state},\n\t fluent_to_swap: {fluent_to_swap},\n\t random_fluent: {random_fluent}, \n\t usable_base_goal_fluents: {usable_base_goal_fluents}\n\n") #debug
                                    
        elif running_recognizability < rec_target[0]:
            #rec too low: choose a goal state that has a fluent from base goal and swap it with a random one
            
            #builds list of fluents that are in the base goal state and also in this goal state
            present_base_goal_fluents = []
            for fluent in base_goal_state:
                if fluent in goal_state:
                    present_base_goal_fluents.append(fluent)
            
            #if there is at least one fluent to swap                    
            if len(present_base_goal_fluents) > 0:
                
                #builds list of fluents that are in the goal state but not in the base goal state, and don't introduce inconsistencies
                # candidates_list = []
                # for fluent in non_base_goal_fluents:
                #     if check_if_fluent_is_usable(fluent, goal_state):
                #         candidates_list.append(fluent)
                
                #build list of possible fluents to use in the swap
                candidates_list = expand_candidates_list(goal_state, base_goal_state, package_for_goal_set, pos_for_goal_set)
                
                #if there is at least one fluent to swap
                if len(candidates_list) > 0:
                    
                    #we will do a while sequence to find a swap that does not introduce inconsistencies, creates duplicates or is already satisfied in the initial state
                    while True:
                        random_fluent = random.choice(candidates_list)

                        fluent_to_swap = random.choice(present_base_goal_fluents)
                        
                        #we create what the state list will look like after the swap
                        proposed_goal_state = goal_state.copy()
                        proposed_goal_state_list = goal_state_list.copy()
                        proposed_goal_state_list[proposed_goal_state_list.index(goal_state)] = proposed_goal_state
                        
                        proposed_goal_state[proposed_goal_state.index(fluent_to_swap)] = random_fluent
                        
                        #then check if it's ok
                        if not check_duplicates(proposed_goal_state_list) and not check_if_goal_state_in_init(initial_state, proposed_goal_state):
                            break
                        else:
                            #if it's not we try another candidate
                            candidates_list.remove(random_fluent)
                            if len(candidates_list) == 0:
                                #print(f"Candidate list is empty because of duplicates, breaking") # debug
                                break
                        #todo some method to expand canditates list if it is empty, will probably need to generate all possible fluents
                        
                    # before = check_consistency(goal_state) #debug
                    
                    #we apply the swap
                    goal_state_list[goal_state_list.index(goal_state)] = proposed_goal_state
                    # after = check_consistency(goal_state) #debug
                    # if before == True and after == False: #debug
                    #     print(f"|<| Goal state is not consistent: {goal_state}\n\t base_goal_state: {base_goal_state},\n\t fluent_to_swap: {fluent_to_swap},\n\t random_fluent: {random_fluent}\n\n") #debug
                    
        #compute the recognizability after the step
        new_recognizability = compute_recognizability(base_goal_state, [base_goal_state] + goal_state_list)
        # Store the regeneration (the key is the recognizability of the prevoius configuration)
        goal_regeneration_dict[running_recognizability] = goal_state_list.copy()
        #if we have not changed the recognizability, the randomness patience is reduced
        #we will try another random goal state in the next iteration 
        if new_recognizability == running_recognizability:
            randomness_patience -= 1
            
            #if we have hit too many times stuck goal states, we will regenerate the last we encountered, so the one in this iteration
            if randomness_patience == 0 and regeneration_patience > 0:
                # Patience reached: regenerate stuck goal_state
                regeneration_patience -= 1
                randomness_patience = randomness_patience_constant
                

                #todo some way to make this not use patience and not loop forever without doing all possible combinations?s
                duplicate_patience = 10
                while duplicate_patience > 0:
                    new_goal_state = generate_goal_state(initial_state, package_for_goal_set, pos_for_goal_set, number_of_goals)
                    
                    proposed_goal_state_list = goal_state_list.copy()
                    proposed_goal_state_list[proposed_goal_state_list.index(goal_state)] = new_goal_state
                    if not check_duplicates(proposed_goal_state_list):
                        break
                    else:
                        duplicate_patience -= 1
                        if duplicate_patience == 0:
                            #print(f"Duplicate patience exhausted, breaking") # debug
                            break
            
                # Replace the goal state in the list with a new one
                goal_state = new_goal_state
            
            #if we have exhausted the regeneration patience, we will stop trying and take the best configuration we have
            elif randomness_patience == 0 and regeneration_patience == 0:
                # Patience exhausted; if we have any regenerations, choose the one with recognizability closest to target.
                if goal_regeneration_dict:
                    # Find the closest recognizability to the target
                    # must use midpoint as we could heve rec_scores both above and below the target
                    target_recognizability = (rec_target[0] + rec_target[1]) / 2 
                    
                    #? if we need to restrict certain type of rec implement these lines
                    while True:
                        closest_recognizability = min(goal_regeneration_dict.keys(), key=lambda x: abs(x - target_recognizability))
                        #we do not want any problem with rec == 1 or 0, hard coded not to accept 
                        if closest_recognizability == 1 or closest_recognizability == 0:
                            goal_regeneration_dict.pop(closest_recognizability)
                        else:
                            break
                    #? else use only this
                    # closest_recognizability = min(goal_regeneration_dict.keys(), key=lambda x: abs(x - target_recognizability))
                    #?
                    
                    running_recognizability = closest_recognizability
                    goal_state_list = goal_regeneration_dict[closest_recognizability]
                #print(f"Patience exhausted, breaking: Best recognizability: {running_recognizability} | Target: {rec_target}") # debug

                # to keep track of errors we do the distace to the target range
                error = abs(running_recognizability - rec_target[0]) if running_recognizability < rec_target[0] else abs(running_recognizability - rec_target[1])
                running_sum_rec_error += error
                global_counter += 1
                break
        #print(f"Running recognizability at end of step: {running_recognizability}")
        running_recognizability = new_recognizability
    return base_goal_state, goal_state_list


class GRProblem:
    def __init__(self, og_problem,  goal_state_list, rec, rec_class=[0,1]):
        
        self.og_problem = og_problem
        self.problem_name = f"{og_problem.name}_{rec_class[0]}-{rec_class[1]}"
        self.objects = og_problem.objects
        self.initial_state = og_problem.init
        self.og_goal_state = og_problem.goal
        self.goal_state_list = goal_state_list
        self.recognizability_class = rec_class
        self.recognizability = rec
    
    def __str__(self):
        description = f"Problem: {self.problem_name}, \n\tOG Goals: {self.og_goal_state}, \n\t"
        for i, goal_state in enumerate(self.goal_state_list):
            description += f"Goal state {i}: {goal_state}, \n\t"
        description += f"Recognizability: {self.recognizability}, Class: {self.recognizability_class}"

        return description
    
    
def generate_dataset(problems, versions_per_plan, rec_classes, plans_to_process=None, test=False):
    gr_problems_list = []
    count = 0
    for problem in problems:
        # control processing limit
        if test and count >= 1:
            break
        if plans_to_process is not None and count > plans_to_process:
            break

        number_of_goals = len(problem.goal)

        # use typed objects directly from problem.objects
        # PDDLProblem.objects maps type names to lists of object strings
        obj_sets = {typ: set(objs) for typ, objs in problem.objects.items()}

        # define which types correspond to our expected categories
        pos_set = obj_sets.get('location', obj_sets.get('pos', set()))
        package_set = obj_sets.get('package', obj_sets.get('obj', set()))

        # goal generation pools (copy to avoid mutation)
        pos_for_goal_set = set(pos_set)
        package_for_goal_set = set(package_set)

        # ensure enough packages for goals
        if number_of_goals > len(package_for_goal_set):
            raise Exception(
                f"Number of goals {number_of_goals} exceeds available packages {len(package_for_goal_set)}")

        # generate and adapt goal states
        for interval in rec_classes:
            goal_state_list = create_goal_state_list(
                initial_state=problem.init,
                package_for_goal_set=package_for_goal_set,
                pos_for_goal_set=pos_for_goal_set,
                number_of_goals=number_of_goals,
                versions_per_plan=versions_per_plan,
            )

            og_goal_state, adapted_goal_state_list = adapt_goal_state_list_to_recognizability(
                initial_state=problem.init,
                base_goal_state=problem.goal,
                goal_state_list=goal_state_list[1:],
                package_for_goal_set=package_for_goal_set,
                pos_for_goal_set=pos_for_goal_set,
                number_of_goals=number_of_goals,
                rec_target=interval,
                randomness_patience=25,
                regeneration_patience=15,
            )

            new_gr_problem = GRProblem(
                og_problem=problem,
                goal_state_list=adapted_goal_state_list,
                rec=compute_recognizability(
                    og_goal_state, [og_goal_state] + adapted_goal_state_list
                ),
                rec_class=interval,
            )
            gr_problems_list.append(new_gr_problem)

        count += 1
    return gr_problems_list


def analyse_dataset(gr_problems_list):
    duplicate_counter = 0
    running_sum_rec_error = 0
    error_counter = 0
    for problem in gr_problems_list:
        if check_duplicates(problem.goal_state_list):
            duplicate_counter += 1
        if (
            problem.recognizability > problem.recognizability_class[1]
            or problem.recognizability < problem.recognizability_class[0]
        ):
            error_counter += 1
            error = (
                abs(problem.recognizability - problem.recognizability_class[0])
                if problem.recognizability < problem.recognizability_class[0]
                else abs(problem.recognizability - problem.recognizability_class[1])
            )
            running_sum_rec_error += error
    print("Error counter:", error_counter)
    print("Running sum of recognizability error:", running_sum_rec_error)
    print(
        "Average recognizability generation error:",
        running_sum_rec_error / error_counter if error_counter > 0 else 0,
    )
    print(
        "Average recognizability generation error on whole dataset:",
        running_sum_rec_error / (plans_to_process * versions_per_plan),
    )

    print("Duplicate counter:", duplicate_counter)
    print("Duplicate percent: %", duplicate_counter / len(gr_problems_list) * 100)
    
dataset_iterations = 3
gr_problem_dict = {}
problems = problems[:50000]
for t in range(0, dataset_iterations):
    print(f"\nStarting dataset {t}:")
    gr_problems_list = generate_dataset(problems=problems, versions_per_plan=versions_per_plan, rec_classes=rec_classes)
    # print("gr_problems_list:")

    analyse_dataset(gr_problems_list)

    gr_problem_dict[t] = gr_problems_list



#check if we have same thing in same orders
print(gr_problem_dict[2][2])
print(gr_problem_dict[1][2])

# runtime with 10k plans, 5 iterations is 4m 30s
#expected runtime at 50k about 25min
#duplicates about 5% per run
#error should be 0.07 or lower for each run

# Example output of generating dataset
# Starting dataset 0:
# Error counter: 83
# Running sum of recognizability error: 4.6932999999999945
# Average recognizability generation error: 0.05654578313253005
# Average recognizability generation error on whole dataset: 1.8773199999999978e-05
# Duplicate counter: 11130
# Duplicate percent: % 4.452

# Starting dataset 1:
# Error counter: 72
# Running sum of recognizability error: 3.8466
# Average recognizability generation error: 0.053425
# Average recognizability generation error on whole dataset: 1.53864e-05
# Duplicate counter: 11170
# Duplicate percent: % 4.468

# Starting dataset 2:
# Error counter: 73
# Running sum of recognizability error: 4.453299999999996
# Average recognizability generation error: 0.06100410958904104
# Average recognizability generation error on whole dataset: 1.7813199999999984e-05
# Duplicate counter: 11191
# Duplicate percent: % 4.4764
# Problem: p014238_0.4-0.6, 
# 	OG Goals: ['at obj12 pos12', 'at obj44 pos21'], 
# 	Goal state 0: ['at obj22 pos22', 'at obj13 pos13'], 
# 	Goal state 1: ['at obj44 pos22', 'at obj66 pos21'], 
# 	Goal state 2: ['at obj12 pos12', 'at obj44 pos13'], 
# 	Goal state 3: ['at obj12 pos13', 'at obj44 pos12'], 
# 	Goal state 4: ['at obj22 pos13', 'at obj44 pos21'], 
# 	Recognizability: 0.4, Class: [0.4, 0.6]
# Problem: p014238_0.4-0.6, 
# 	OG Goals: ['at obj12 pos12', 'at obj44 pos21'], 
# 	Goal state 0: ['at obj12 pos12', 'at obj66 pos12'], 
# 	Goal state 1: ['at obj13 pos33', 'at obj12 pos12'], 
# 	Goal state 2: ['at obj13 pos13', 'at obj44 pos12'], 
# 	Goal state 3: ['at obj12 pos33', 'at obj44 pos33'], 
# 	Goal state 4: ['at obj13 pos77', 'at obj22 pos21'], 
# 	Recognizability: 0.6, Class: [0.4, 0.6]


# we want as little as possible duplicates, so we generate the dataset N times 
# then we see if we can intersect them into a single dataset, if we have more than one candidate we take the one closer to target rec class
# these could be easily modified to expand the dataset, 
# we could instead of taking original goal state, generate a random one also there, keeping only number of fluent in goal the same
final_gr_problem_list = []
for i in range(0, len(gr_problem_dict[0])):
    versions = []
    candidates = []
    for j in range(0, dataset_iterations):
        versions.append(gr_problem_dict[j][i])
        
    for problem in versions:
        if not check_duplicates(problem.goal_state_list):
            candidates.append(problem)

    if candidates == []:
        candidates = versions

    candidates_error_dict = {}
    
    target = (versions[0].recognizability_class[1]+versions[0].recognizability_class[0])/2
    for problem in candidates:
        error = error = (
            abs(problem.recognizability - target)
            )
        candidates_error_dict[error] = problem
        
    min_error_problem = candidates_error_dict[min(candidates_error_dict.keys())]
    final_gr_problem_list.append(min_error_problem)

analyse_dataset(final_gr_problem_list)



# error still high when it happens, happens less, maybe duplicates introduced error because they were stuck, duplicates are pretty much insignificant

base_goal_in_init_counter = 0
versions_goals_in_init_counter = 0
problems_to_drop = []
for problem in final_gr_problem_list:
    base_goal_present_in_init = True
    for base_goal_fluent in problem.og_goal_state:
        # print(base_goal_fluent)
        # print(problem.og_plan.initial_state)
        if base_goal_fluent in problem.initial_state:
            continue
        else:
            base_goal_present_in_init = False
            break
    
    if base_goal_present_in_init:
        base_goal_in_init_counter += 1
        # print(f"Base Goal present in init: {problem.og_goal_state} \n Problem init: {problem.initial_state}")
        
    for goal_state in problem.goal_state_list:
        version_goal_present_in_init = True
        for goal_fluent in goal_state:
            if goal_fluent in problem.initial_state:
                # print(f"Partial match: {goal_state} \n Problem init: {problem.og_plan.initial_state}")
                continue
            else:
                version_goal_present_in_init = False
        if version_goal_present_in_init:
            # print(f"Problem {problem.problem_name}")
            # print(f"Version Goal present in init: {goal_state} \n Problem init: {problem.initial_state}")
            problems_to_drop.append(problem)
            versions_goals_in_init_counter += 1

perc = round((base_goal_present_in_init/len(gr_problems_list)*100),3)
print(f"Original goal state that are already true in init: {base_goal_present_in_init} ({perc}%)")

perc = round((versions_goals_in_init_counter/len(gr_problems_list)*100),3)
print(f"Versions goal state that are already true in init: {versions_goals_in_init_counter} ({perc} %)")


begin_len = len(final_gr_problem_list)

final_gr_problem_list_after_drop = []
for problem in final_gr_problem_list:
    if check_duplicates(problem.goal_state_list):
        # print(f"Problem dropped: {problem.problem_name},{problem.recognizability_class}")
        continue
    elif (
        problem.recognizability > problem.recognizability_class[1]
        or problem.recognizability < problem.recognizability_class[0]
    ):
        continue
    else:
        if not problem in problems_to_drop:
            final_gr_problem_list_after_drop.append(problem)           
end_len = len(final_gr_problem_list_after_drop)


diff = begin_len - end_len
perc = diff/begin_len*100
print(f"Dropped: {begin_len}-{end_len} = {diff}({perc}%)")


from collections import defaultdict

desired_problems = 50000 #adjust number as needed
# First, deduplicate problems by name so that each problem (from the same plan) appears only once

problem_groups = defaultdict(list)
for problem in final_gr_problem_list_after_drop:
    problem_groups[problem.problem_name].append(problem)

unique_problems = {name: random.choice(problems) for name, problems in problem_groups.items()}

# Group unique problems by recognizability class (using tuple of bounds as key)
problems_by_class = defaultdict(list)
for problem in unique_problems.values():
    rec_class_key = tuple(problem.recognizability_class)
    problems_by_class[rec_class_key].append(problem)

# Number of recognizability classes, assuming rec_classes defined in cell 1
num_classes = len(rec_classes)
desired_per_class = desired_problems // num_classes 

chosen_problems = []
for rec_class_key, problems in problems_by_class.items():
    if len(problems) >= desired_per_class:
        chosen_problems.extend(random.sample(problems, desired_per_class))
    else:
        chosen_problems.extend(problems)
        

print("Uniformly sampled problems (initial):", len(chosen_problems))

# If the total is less than 1000, fill with additional problems randomly from the remaining ones
if len(chosen_problems) < desired_problems:
    remaining = desired_problems - len(chosen_problems)
    remaining_pool = [p for p in final_gr_problem_list_after_drop if p not in chosen_problems]
    if remaining_pool:
        extra = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
        chosen_problems.extend(extra)
    print("After extra sampling, total problems:", len(chosen_problems))
    
    
import matplotlib.pyplot as plt

# Build a distribution dictionary with keys as "low-high" strings for each recognizability class
distribution = {f"{rc[0]}-{rc[1]}": 0 for rc in rec_classes}
for problem in chosen_problems:
    key = f"{problem.recognizability_class[0]}-{problem.recognizability_class[1]}"
    distribution[key] += 1

# Plot the distribution as a bar graph
plt.figure(figsize=(8, 4))
plt.bar(distribution.keys(), distribution.values(), color='skyblue')
plt.xlabel("Recognizability Class Interval")
plt.ylabel("Number of Problems")
plt.title("Distribution of Recognizability Classes")
plt.show()


def generate_problem_string(pddl_problem, goal_state):
    # Use regex to extract the plan name (e.g., p034533) from strings like "p034533_5" or "p034533_og"
    
    #definition
    problem_string = ""
    
    problem_string += f"(define (problem {pddl_problem.name}_og)\n(:domain {domain})\n(:objects\n\t"

    
    #objects in a dict format, {type: obj_set}
    for type, obj_set in pddl_problem.objects.items():
        if len(obj_set) > 0:
            for obj in obj_set:
                problem_string += f"{obj} "
            problem_string += f"- {type}\n\t"
    problem_string += f")\n"
    
    #initial state
    problem_string += f"(:init\n"
    for fluent in pddl_problem.init:
        problem_string += f"\t({fluent})\n"
    problem_string += f")\n"
    
    #goal state
    problem_string += f"(:goal (and\n"
    for goal in goal_state:
        problem_string += f"\t({goal})\n"
    problem_string += f"))\n)"
    
    return problem_string


def write_problem(problem):

    out_dir = f"{results_dir}/{problem.recognizability_class[0]}-{problem.recognizability_class[1]}/{problem.og_problem.name}"
    os.makedirs(out_dir, exist_ok=True)
    base_problem_file = f"{out_dir}/{problem.og_problem.name}_og.pddl"
    base_problem_text = generate_problem_string(problem.og_problem, problem.og_goal_state)
    with open(base_problem_file, "w") as f:
        f.write(base_problem_text)
        
    for i,goal_state in enumerate(problem.goal_state_list):
        print(goal_state)
        version_text = generate_problem_string(pddl_problem=problem.og_problem, goal_state=goal_state)
        version_problem_file = f"{out_dir}/{problem.og_problem.name}_{i}.pddl"

        with open(version_problem_file, "w") as f:
            f.write(version_text)
            
            
print(len(chosen_problems))
for problem in chosen_problems:
    write_problem(problem)

    # found = False
    # for sol in os.listdir(sol_dir):
    #     if sol.startswith(f"{problem.name}"):
    #         solutions_to_og_problems.append(sol)
    #         found = True
    #         break
    # if not found:
    #     print(f"Sol not found for {problem.name}")
