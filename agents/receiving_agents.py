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
        pass

    def process_signal(self):
        highest_intention: int = 0
        highest_index: int = 0

        for i, intention in enumerate(self.intentions):
            if intention > highest_intention:
                highest_intention = intention
                highest_index = i
        
        # Make random choice through epsilon probability
        if check_epsilon(EPS):
            return make_random_choice()

        return highest_index

    def update(self, action: int):
        # Update beliefs

        # Update intentions
        acc: float = sum(self.intentions)
        acc = (1.0 - LEARNING_SPEED) / acc

        for intention in self.intentions:
            intention = intention * acc

        self.intentions[action] = self.intentions[action] + LEARNING_SPEED


