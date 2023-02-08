from abc import ABC, abstractmethod
from utilities import generate_beliefs, generate_2d_beliefs

LEARNING_SPEED = 0.1 
EPS = 0.1 # Epsilon value to ensure semi-stochastic decision making

class TheoryOfMindAgent(ABC):
    '''
    Base Theory of Mind Agent abstract 
    class for type hinting.
    '''
    def __init__(self, beliefs = generate_beliefs(), intentions = generate_beliefs()) -> None:
        self.beliefs = beliefs
        self.intentions = intentions

    @abstractmethod
    def decide(self, **kwargs) -> int:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        pass

class SignalingAgent(TheoryOfMindAgent):
    '''
    Base Theory of Mind Signaling Agent 
    abstract class for type hinting.
    '''
    def __init__(self, beliefs=generate_2d_beliefs(), intentions=generate_beliefs()) -> None:
        self.chosen_signal = 0
        super().__init__(beliefs, intentions)

    @abstractmethod
    def signal(self, **kwargs) -> int:
        pass

class ReceivingAgent(TheoryOfMindAgent):
    '''
    Base Theory of Mind Receiving Agent 
    abstract class for type hinting.
    '''
    def __init__(
        self, 
        beliefs=generate_beliefs(), 
        intentions=generate_beliefs(), 
        connected_beliefs=generate_2d_beliefs()
    ) -> None:
        super().__init__(beliefs, intentions)
        self.connected_beliefs = connected_beliefs

    @abstractmethod
    def process_signal(self, signal: int, **kwargs) -> None:
        pass