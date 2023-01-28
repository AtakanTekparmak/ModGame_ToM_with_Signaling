from abc import ABC, abstractmethod
from utilities import generate_beliefs

LEARNING_SPEED = 0.1 
EPS = 0.1 # Epsilon value to ensure semi-stochastic decision making

class TheoryOfMindAgent(ABC):

    def __init__(self, beliefs = generate_beliefs(), intentions = generate_beliefs()) -> None:
        self.beliefs = beliefs
        self.intentions = intentions

    @abstractmethod
    def decide(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass