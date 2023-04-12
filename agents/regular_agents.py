from utilities import generate_beliefs, check_epsilon, make_random_choice
from agents.agent import TheoryOfMindAgent, EPS, LEARNING_SPEED

class ZeroOrderTheoryOfMindAgent(TheoryOfMindAgent):
    '''
    Zero-order Theory of Mind Agent. Has simple 1D beliefs
    and makes decisions based on the highest belief.
    '''
    def __init__(self, beliefs = generate_beliefs(), intentions = generate_beliefs()) -> None:
        super().__init__(beliefs, intentions)

    def decide(self):
        highest_belief: int = 0
        highest_index: int = 0

        for i, belief in enumerate(self.beliefs):
            if belief > highest_belief:
                highest_belief = belief
                highest_index = i
        
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            return make_random_choice()

        return (highest_index + 1) % 23
    
    def update(self, action: int):
        acc: float = sum(self.beliefs)
        acc = (1.0 - LEARNING_SPEED) / acc

        for belief in self.beliefs:
            belief = belief * acc

        self.beliefs[action] = self.beliefs[action] + LEARNING_SPEED


class FirstOrderTheoryOfMindAgent(TheoryOfMindAgent):
    '''
    First-order Theory of Mind Agent. Acts greedily 
    based on a model of the zero-order agent's decision-making process.
    '''
    def __init__(self, lower_order_agent: TheoryOfMindAgent = ZeroOrderTheoryOfMindAgent()) -> None:
        self.zero_order_agent = lower_order_agent

    def decide(self):
        # Model the decision-making process of the zero-order agent
        zero_order_decision = self.zero_order_agent.decide()

        # Make a decision based on the zero-order decision
        return (zero_order_decision + 1) % 23
    
    def update(self, action: int):
        # Update the beliefs of the zero-order agent
        self.zero_order_agent.update(action)

class SecondOrderTheoryOfMindAgent(TheoryOfMindAgent):
    '''
    Second-order Theory of Mind Agent. Acts greedily on
    a model of the zero-order and first-order agents' 
    decision-making processes. Has beliefs for which
    order of agent is more likely to be present.
    '''
    def __init__(
        self, 
        zero_order_agent: ZeroOrderTheoryOfMindAgent = ZeroOrderTheoryOfMindAgent(),
        first_order_agent: FirstOrderTheoryOfMindAgent = FirstOrderTheoryOfMindAgent()
        ) -> None:
        self.order_beliefs = generate_beliefs(2)
        self.zero_order_agent = zero_order_agent
        self.first_order_agent = first_order_agent

    def decide(self):
        if check_epsilon(EPS): # Epsilon for stochasticity 
            order: int = make_random_choice(2)
            if order == 0:
                return (self.zero_order_agent.decide() + 1) % 23  
            else:
                return (self.first_order_agent.decide() + 1) % 23
        else:
            if self.order_beliefs[0] > self.order_beliefs[1]: # More zero order agents
                return (self.zero_order_agent.decide() + 1) % 23
            else: # More first order agents
                return (self.first_order_agent.decide() + 1) % 23

    def update(self, action: int):
        # Get each lower level agent's decision and the respective higher order decision
        zero_order_decision: int = self.zero_order_agent.decide()
        first_order_decision: int = self.first_order_agent.decide()
        zero_order_higher_decision: int = (zero_order_decision + 1) % 23
        first_order_higher_decision: int = (first_order_decision + 1) % 23

        # Update the beliefs of the zero and first-order agents
        self.zero_order_agent.update(action)
        self.first_order_agent.update(action)

        # Update order beliefs
        if action == zero_order_higher_decision or action == first_order_higher_decision:
            acc: float = sum(self.order_beliefs)
            acc = (1.0 - LEARNING_SPEED) / acc

            for belief in self.order_beliefs:
                belief = belief * acc

            if action == zero_order_higher_decision:
                self.order_beliefs[0] = self.order_beliefs[0] + LEARNING_SPEED
            else:
                self.order_beliefs[1] = self.order_beliefs[1] + LEARNING_SPEED