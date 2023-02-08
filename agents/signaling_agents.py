from agents.agent import SignalingAgent, EPS, LEARNING_SPEED
from agents.regular_agents import ZeroOrderTheoryOfMindAgent
from utilities import generate_beliefs, generate_2d_beliefs, check_epsilon, make_random_choice

class ZeroOrderSignalingAgent(SignalingAgent):
    def __init__(self, beliefs=generate_2d_beliefs(), intentions=generate_beliefs()) -> None:
        super().__init__(beliefs, intentions)

    def signal(self):
        highest_signal: int = 0
        highest_index: int = 0

        for i, signals in enumerate(self.beliefs):
            if sum(signals) > highest_signal:
                highest_signal = signals
                highest_index = i
        
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            choice: int = make_random_choice()
            self.chosen_signal = choice
            return choice

        return highest_index

    def decide(self, **kwargs) -> int:
        highest_decision: int = 0
        highest_index: int = 0

        for i, action in enumerate(self.beliefs[self.chosen_signal]):
            if action > highest_decision:
                highest_decision = action
                highest_index = i

        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            choice: int = make_random_choice()
            self.chosen_signal = choice
            return choice

        return highest_index

    def update(self, action: int):
        # Update beliefs
        chosen_action: int = self.decide()

        if chosen_action == (action + 1) % 23:
            # Update beliefs
            for sig_index, signal in enumerate(self.beliefs):
                acc: float = sum(signal)
                acc = (1.0 - LEARNING_SPEED) / acc

                for decision in signal:
                    decision = decision * acc

                    self.beliefs[sig_index][chosen_action] = self.beliefs[sig_index][chosen_action] + LEARNING_SPEED


