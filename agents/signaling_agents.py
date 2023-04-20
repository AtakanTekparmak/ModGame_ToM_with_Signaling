from agents.agent import SignalingAgent, EPS, LEARNING_SPEED
from agents.regular_agents import ZeroOrderTheoryOfMindAgent
from utilities import generate_beliefs, generate_2d_beliefs, check_epsilon, make_random_choice

class ZeroOrderSignalingAgent(SignalingAgent):
    '''
    Zero-order Signaling Agent. Acts greedily based on 
    the highest belief in the 2D beliefs array. Updates
    beliefs based on the chosen signal and action.
    '''
    def __init__(self, beliefs=generate_2d_beliefs(), intentions=generate_beliefs()) -> None:
        super().__init__(beliefs, intentions)

    def signal(self):
        highest_signal: int = 0
        highest_index: int = 0

        for i, signals in enumerate(self.beliefs):
            sum_signals = sum(signals)
            if sum_signals > highest_signal:
                highest_signal = sum_signals
                highest_index = i
        
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            choice: int = make_random_choice()
            self.chosen_signal = choice
            return choice

        self.chosen_signal = highest_index
        return highest_index

    def decide(self, **kwargs) -> int:
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            choice: int = make_random_choice()
            self.chosen_action = choice
            return choice
        
        highest_decision: int = 0
        highest_index: int = 0

        for i, action in enumerate(self.beliefs[self.chosen_signal]):
            if action > highest_decision:
                highest_decision = action
                highest_index = i

        self.chosen_action = highest_index
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


class FirstOrderSignalingAgent(SignalingAgent):
    '''
    First-order Theory of Mind Signaling Agent. Acts greedily based on 
    a model of the zero-order agent's decision-making process. Updates
    beliefs based on the chosen signal and action.
    '''
    def __init__(
            self, 
            beliefs=generate_2d_beliefs(),
            lower_order_agent: ZeroOrderSignalingAgent = ZeroOrderSignalingAgent()
        ) -> None:
        super().__init__(generate_2d_beliefs(), generate_beliefs())
        self.zero_order_agent = lower_order_agent

    def signal(self, **kwargs) -> int:
        # Model the decision-making process of the zero-order agent
        zero_order_decision = self.zero_order_agent.signal()

        self.chosen_signal = zero_order_decision
        return zero_order_decision

    def decide(self, **kwargs) -> int:
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            choice: int = make_random_choice()
            self.chosen_action = choice
            return choice

        highest_decision: int = 0
        highest_index: int = 0

        for i, action in enumerate(self.beliefs[self.chosen_signal]):
            if action > highest_decision:
                highest_decision = action
                highest_index = i

        self.chosen_action = highest_index
        return highest_index

    def update(self, action: int):
        # Update the beliefs of the zero-order agent
        self.zero_order_agent.update(action)

        # Update beliefs
        chosen_action: int = self.decide()

        if chosen_action == (action + 1) % 23:
            for signal in range(len(self.beliefs)):
                for decision in range(len(self.beliefs[signal])):
                    if signal == self.chosen_signal and decision == chosen_action:
                        self.beliefs[signal][decision] += LEARNING_SPEED * (1 - self.beliefs[signal][decision])
                    else:
                        self.beliefs[signal][decision] *= (1 - LEARNING_SPEED)


class SecondOrderSignalingAgent(SignalingAgent):
    '''
    Second-order Theory of Mind Signaling Agent. Acts greedily on
    a model of the zero-order and first-order agents' decision-making
    processes. Updates beliefs based on the chosen signal and action.
    '''
    def __init__(
        self, 
        zero_order_agent: ZeroOrderSignalingAgent = ZeroOrderSignalingAgent(),
        first_order_agent: FirstOrderSignalingAgent = FirstOrderSignalingAgent(),
    ) -> None:
        super().__init__(generate_2d_beliefs(), generate_beliefs())
        self.order_beliefs = generate_beliefs(2)
        self.zero_order_agent = zero_order_agent
        self.first_order_agent = first_order_agent

    def signal(self, **kwargs) -> int:
        # Model the decision-making process of the zero and first-order agents
        if check_epsilon(EPS):
            order: int = make_random_choice(2)
            if order == 0:
                self.chosen_signal = self.zero_order_agent.signal()
            else:
                self.chosen_signal = self.first_order_agent.signal()
        else:
            if self.order_beliefs[0] > self.order_beliefs[1]:
                self.chosen_signal = self.zero_order_agent.signal()
            else:
                self.chosen_signal = self.first_order_agent.signal()

        return self.chosen_signal

    def decide(self, **kwargs) -> int:
        # Model the decision-making process of the zero and first-order agents
        if check_epsilon(EPS):
            order: int = make_random_choice(2)
            if order == 0:
                self.chosen_action = self.zero_order_agent.decide()
            else:
                self.chosen_action = self.first_order_agent.decide()
        else:
            if self.order_beliefs[0] > self.order_beliefs[1]:
                self.chosen_action = self.zero_order_agent.decide()
            else:
                self.chosen_action = self.first_order_agent.decide()

        return self.chosen_action

    def update(self, action: int):
        # Get each lower level agent's decision and the respective higher order decision
        zero_order_decision: int = self.zero_order_agent.decide()
        first_order_decision: int = self.first_order_agent.decide()

        # Update the beliefs of the zero and first-order agents
        self.zero_order_agent.update(action)
        self.first_order_agent.update(action)

        # Update order beliefs
        if action == (zero_order_decision + 1) % 23 or action == (first_order_decision + 1) % 23:
            acc: float = sum(self.order_beliefs)
            acc = (1.0 - LEARNING_SPEED) / acc

            for belief in self.order_beliefs:
                belief = belief * acc

            if action == (zero_order_decision + 1) % 23:
                self.order_beliefs[0] = self.order_beliefs[0] + LEARNING_SPEED
            else:
                self.order_beliefs[1] = self.order_beliefs[1] + LEARNING_SPEED
        