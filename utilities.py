import numpy as np
from dataclasses import dataclass

NUM_OF_CHOICES = 23

@dataclass
class AgentsConfiguration:
    '''
    Utility dataclass to store 
    the number of each level
    of Theory of Mind agent.
    '''
    zero_order_agent_number: int
    first_order_agent_number: int
    second_order_agent_number: int

def generate_beliefs(number_of_choices: int = NUM_OF_CHOICES):
    ''' Generates a set of random beliefs. '''
    random_numpy_list = list(np.random.rand(number_of_choices,1))
    new_list = []

    for element in random_numpy_list:
        new_list.append(list(element))

    return [item for sublist in new_list for item in sublist]

def check_epsilon(eps: float):
    ''' 
    Samples a random float between 0.0 and 
    1.0 to check against an epsilon value.
    '''
    probability: float = np.random.random_sample()
    return probability < eps

def make_random_choice(number_of_choices: int = NUM_OF_CHOICES):
    ''' Makes a random choice given the number of choices. '''
    return np.random.randint(number_of_choices)