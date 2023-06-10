from agents.agent import ReceivingAgent, EPS, LEARNING_SPEED
from utilities import generate_beliefs, generate_2d_beliefs, check_epsilon, make_random_choice

class ZeroOrderReceivingAgent(ReceivingAgent):
    '''
    Zero-order Receiving Agent. 
    '''
    def __init__(
        self, 
        beliefs=generate_beliefs(), 
        intentions=generate_beliefs(), 
        connected_beliefs= generate_2d_beliefs()
        ) -> None:
        super().__init__(beliefs, intentions, connected_beliefs)

    def decide(self):
        """ Decide on an action based on beliefs """
        # Get the most prominent belief index
        belief_index = self.beliefs.index(max(self.beliefs))

        # Check if agent should act randomly
        if check_epsilon(EPS):
            return make_random_choice()

        # Return the action with the highest connected belief
        return self.connected_beliefs[belief_index].index(max(self.connected_beliefs[belief_index]))

    def process_signal(self, signal: int):
        """ Update beliefs based on signal """
        # Set learning speed to 0.4 for faster convergence, as beliefs are reset every round
        LEARNING_SPEED = 0.4

        # Update beliefs
        acc: float = sum(self.beliefs)
        acc = (1.0 - LEARNING_SPEED) / acc

        for belief in self.beliefs:
            belief = belief * acc

        self.beliefs[signal] = self.beliefs[signal] + LEARNING_SPEED

    def update(self, action: int):
        """ Update connected beliefs based on action """
        # Get the most prominent belief index
        belief_index = self.beliefs.index(max(self.beliefs))

        # Update connected beliefs
        acc: float = sum(self.connected_beliefs[belief_index])
        acc = (1.0 - LEARNING_SPEED) / acc

        for belief in self.connected_beliefs[belief_index]:
            belief = belief * acc

        self.connected_beliefs[belief_index][action] = self.connected_beliefs[belief_index][action] + LEARNING_SPEED

        # Reset beliefs
        self.beliefs = generate_beliefs()

class FirstOrderReceivingAgent(ReceivingAgent):
    '''
    First-order Receiving Agent. 
    '''
    def __init__(
        self, 
        beliefs=generate_beliefs(), 
        lower_order_agent: ZeroOrderReceivingAgent = ZeroOrderReceivingAgent(),
        ) -> None:
        super().__init__(beliefs)
        self.lower_order_agent = lower_order_agent

    def decide(self):
        """ Decide on an action based on lower order agent"""
        zero_order_decision = self.lower_order_agent.decide()

        # Check if agent should act randomly
        if check_epsilon(EPS):
            return make_random_choice()
        
        return (zero_order_decision + 1) % 23
    
    def process_signal(self, signal: int):
        """ Update beliefs based on signal """
        self.lower_order_agent.process_signal(signal)

    def update(self, action: int):
        """ Update connected beliefs based on action """
        self.lower_order_agent.update(action)

class SecondOrderReceivingAgent(ReceivingAgent):
    '''
    Second-order Receiving Agent. 
    '''
    def __init__(
            self, 
            zero_order_agent: ZeroOrderReceivingAgent = ZeroOrderReceivingAgent(),
            first_order_agent: FirstOrderReceivingAgent = FirstOrderReceivingAgent(),
        ) -> None:
        self.order_beliefs = generate_beliefs(2)
        self.zero_order_agent = zero_order_agent
        self.first_order_agent = first_order_agent

    def decide(self):
        order: int = 0

        if check_epsilon(EPS):
            order = make_random_choice()
        else:
            if self.order_beliefs[0] > self.order_beliefs[1]:
                order = 0
            else:
                order = 1

        if order == 0:
            return (self.zero_order_agent.decide() + 1) % 23
        else:
            return (self.first_order_agent.decide() + 1) % 23
        
    def process_signal(self, signal: int):
        """ Update beliefs based on signal """
        self.zero_order_agent.process_signal(signal)
        self.first_order_agent.process_signal(signal)

    def update(self, action: int):
        zero_order_decision = self.zero_order_agent.decide()
        first_order_decision = self.first_order_agent.decide()

        # Update beliefs of lower order agents
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