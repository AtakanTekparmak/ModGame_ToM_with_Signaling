from abc import ABC, abstractmethod
from utilities import AgentsConfiguration

class Simulation(ABC):
    ''' Abstract class to define simulations. '''
    def __init__(self, agent_config: AgentsConfiguration) -> None:
        self.agent_config = agent_config

    def run(self, number_of_epochs: int = 1000):
        '''
        Runs the simulation for a specified 
        number of epochs/rounds.
        '''
        for _ in range(number_of_epochs):
            self.simulate_round()

    @abstractmethod
    def simulate_round(self, **kwargs):
        pass

    @abstractmethod
    def display_results(self, **kwargs):
        '''
        Displays the results of the simulation.
        '''
        pass