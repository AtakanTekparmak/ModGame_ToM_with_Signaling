import numpy as np
from dataclasses import dataclass
from typing import Optional

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

@dataclass
class SignalingAgentsConfiguration:
    '''
    Utility dataclass to store
    the number of each level
    of Theory of Mind agent.
    '''
    zero_order_sig_agent_number: int
    zero_order_agent_number: int
    first_order_sig_agent_number: int
    first_order_agent_number: int
    second_order_sig_agent_number: int
    second_order_agent_number: int

@dataclass
class RegularSimulationResults:
    '''
    Utility dataclass to 
    store agent results.
    '''
    # Zero order results
    zero_order_mean: float
    zero_order_std: float 
    first_order_mean: float
    first_order_std: float 
    second_order_mean: float
    second_order_std: float 
    extras_mean: Optional[float] = 0.0
    extras_std: Optional[float] = 0.0

@dataclass
class SignalingSimulationResults:
    '''
    Utility dataclass to
    store signaling agent results.
    '''
    zero_order_sig_mean: float
    zero_order_sig_std: float
    zero_order_mean: float
    zero_order_std: float
    first_order_sig_mean: float
    first_order_sig_std: float
    first_order_mean: float
    first_order_std: float
    second_order_sig_mean: float
    second_order_sig_std: float
    second_order_mean: float
    second_order_std: float


def generate_beliefs(number_of_choices: int = NUM_OF_CHOICES):
    ''' Generates a set of random beliefs. '''
    random_numpy_list = list(np.random.rand(number_of_choices,1))
    new_list = []

    for element in random_numpy_list:
        new_list.append(list(element))

    return [item for sublist in new_list for item in sublist]

def generate_2d_beliefs(number_of_choices: int = NUM_OF_CHOICES):
    ''' Generates a set of random beliefs for receiving agents. '''
    random_numpy_list = list(np.random.rand(number_of_choices,number_of_choices))
    new_list = []

    for element in random_numpy_list:
        new_list.append(list(element))

    return new_list

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

def get_mean(array_like) -> float:
    ''' Gets the mean of an array-like (list, np array, etc.)'''
    return np.mean(array_like)

def get_std(array_like) -> float:
    ''' Gets the standard deviation of an array-like (list, np array, etc.)'''
    return np.std(array_like)