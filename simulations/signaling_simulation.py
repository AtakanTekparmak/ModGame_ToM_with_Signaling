from typing import List

from agents.signaling_agents import (
    ZeroOrderSignalingAgent,
    FirstOrderSignalingAgent,
    SecondOrderSignalingAgent
)
from agents.regular_agents import (
    ZeroOrderTheoryOfMindAgent,
    FirstOrderTheoryOfMindAgent,
    SecondOrderTheoryOfMindAgent
)
from agents.receiving_agents import (
    ZeroOrderReceivingAgent,
    FirstOrderReceivingAgent,
    SecondOrderReceivingAgent
)
from agents.agent import SignalingAgent, ReceivingAgent
from utilities import (
    SignalingAgentsConfiguration,
    SignalingSimulationResults,
    RegularSimulationResults, 
    get_mean,
    get_std
)
from simulations.simulation import Simulation
        
def create_signaling_agents(
        agent_config: SignalingAgentsConfiguration = SignalingAgentsConfiguration(10, 10, 10, 10, 10, 10)
    ) -> List[SignalingAgent]:
    ''' 
    Creates a set population of agents based on 
    an SignalingAgentsConfiguration instance.

    Args:
        agent_config: SignalingAgentsConfiguration == The Agent Population Config
    '''
    # Extract agent numbers
    number_0 = agent_config.zero_order_sig_agent_number
    number_2 = agent_config.first_order_sig_agent_number
    number_4 = agent_config.second_order_sig_agent_number

    # Create a list of zero-order agents
    zero_order_agents = [ZeroOrderSignalingAgent() for _ in range(number_0)]

    # Create a list of first-order agents, each with a reference to a zero-order agent
    first_order_agents = [FirstOrderSignalingAgent() for _ in range(number_2)]

    # Create a list of second-order agents, each with a reference to a zero and first-order agent
    second_order_agents = [SecondOrderSignalingAgent() for _ in range(number_4)]

    # Create a list of all agents
    return zero_order_agents + first_order_agents + second_order_agents

def create_receiving_agents(
        agent_config: SignalingAgentsConfiguration = SignalingAgentsConfiguration(10, 10, 10, 10, 10, 10)
    ) -> List[ReceivingAgent]:
    '''
    Creates a set population of agents based on
    an SignalingAgentsConfiguration instance.

    Args:
        agent_config: SignalingAgentsConfiguration == The Agent Population Config
    '''

    # Extract agent numbers
    number_0 = agent_config.zero_order_rec_agent_number
    number_1 = agent_config.first_order_rec_agent_number
    number_2 = agent_config.second_order_rec_agent_number

    # Create a list of zero-order agents
    zero_order_agents = [ZeroOrderReceivingAgent() for _ in range(number_0)]

    # Create a list of first-order agents
    first_order_agents = [FirstOrderReceivingAgent() for _ in range(number_1)]

    # Create a list of second-order agents
    second_order_agents = [SecondOrderReceivingAgent() for _ in range(number_2)]

    return zero_order_agents + first_order_agents + second_order_agents


class SignalingSimulation(Simulation):
    '''
    A Simulation of Theory of Mind
    agents playing the mod game with
    23 choices, with signaling.
    '''
    def __init__(
        self,
        agent_config: SignalingAgentsConfiguration = SignalingAgentsConfiguration(10, 10, 10, 10, 10, 10)
    ):
        super().__init__(agent_config)
        self.signaling_agents: List[SignalingAgent] = create_signaling_agents(agent_config)
        self.receiving_agents: List[ReceivingAgent] = create_receiving_agents(agent_config)
        #self.signaling_agent_actions = [[] for _ in self.signaling_agents]
        self.signaling_agent_scores = [0 for _ in self.signaling_agents]
        self.receiving_agent_scores = [0 for _ in self.receiving_agents]

    def simulate_round(self) -> None:
        # Have signaling agents decide on a signal
        signals = [agent.signal() for agent in self.signaling_agents if isinstance(agent, SignalingAgent)]

        # Have receiving agents process the signals
        for agent in self.receiving_agents:
            for signal in signals:
                agent.process_signal(signal)
        
        # Have agents decide on an action
        actions = [agent.decide() for agent in self.signaling_agents]
        signaling_length: int = len(actions)

        actions = actions + [agent.decide() for agent in self.receiving_agents]
        actions_copy = actions

        for index, action in enumerate(actions):
            lower_choices = sum([1 for choice in actions_copy if choice == (action - 1) % 23])

            if index < signaling_length:
                self.signaling_agent_scores[index] += lower_choices
            else:
                self.receiving_agent_scores[index - signaling_length] += lower_choices

        # Update the beliefs of signaling agents based on the actions of all agents
        for agent, action in zip(self.signaling_agents, actions):
            agent.update(action)

        # Update the beliefs of receiving agents based on the actions of all agents
        for agent, action in zip(self.receiving_agents, actions):
            agent.update(action)

    def get_results(self) -> RegularSimulationResults:
        '''
        Calculates the statistics and returns 
        them in a dict with individual results.
        '''
        # Extract agent numbers
        number_0 = self.agent_config.zero_order_sig_agent_number
        number_1 = number_0 + self.agent_config.first_order_sig_agent_number
        number_2 = number_1 + self.agent_config.second_order_sig_agent_number
        number_3 = self.agent_config.zero_order_rec_agent_number
        number_4 = number_3 + self.agent_config.first_order_rec_agent_number
        number_5 = number_4 + self.agent_config.second_order_rec_agent_number

        # Slice scores per ToM level
        # Get signaling agent scores
        zero_order_sig_scores = self.signaling_agent_scores[:number_0]
        first_order_sig_scores = self.signaling_agent_scores[number_0:number_1]
        second_order_sig_scores = self.signaling_agent_scores[number_1:number_2]

        # Get receiving agent scores
        zero_order_rec_scores = self.receiving_agent_scores[:number_3]
        first_order_rec_scores = self.receiving_agent_scores[number_3:number_4]
        second_order_rec_scores = self.receiving_agent_scores[number_4:number_5]

        return SignalingSimulationResults(
            zero_order_sig_mean=get_mean(zero_order_sig_scores),
            zero_order_sig_std=get_std(zero_order_sig_scores),
            first_order_sig_mean=get_mean(first_order_sig_scores),
            first_order_sig_std=get_std(first_order_sig_scores),
            second_order_sig_mean=get_mean(second_order_sig_scores),
            second_order_sig_std=get_std(second_order_sig_scores),
            zero_order_rec_mean=get_mean(zero_order_rec_scores),
            zero_order_rec_std=get_std(zero_order_rec_scores),
            first_order_rec_mean=get_mean(first_order_rec_scores),
            first_order_rec_std=get_std(first_order_rec_scores),
            second_order_rec_mean=get_mean(second_order_rec_scores),
            second_order_rec_std=get_std(second_order_rec_scores),
        )

    def display_results(self, results: SignalingSimulationResults) -> None:

        print("Printing statistics ...")
        print("=====================================")
        print("Signaling Agents Statistics")
        print("=====================================")
        print(f"Zero Order Signaling Mean Score: {results.zero_order_sig_mean:.3f}")
        print(f"Zero Order Signaling Std: {results.zero_order_sig_std:.3f}")
        print(f"First Order Signaling Mean Score: {results.first_order_sig_mean:.3f}")
        print(f"First Order Signaling Std: {results.first_order_sig_std:.3f}")
        print(f"Second Order Signaling Mean Score: {results.second_order_sig_mean:.3f}")
        print(f"Second Order Signaling Std: {results.second_order_sig_std:.3f}")
        print("=====================================")
        print("Receiving Agents Statistics")
        print("=====================================")
        print(f"Zero Order Receiving Mean Score: {results.zero_order_rec_mean:.3f}")
        print(f"Zero Order Receiving Std: {results.zero_order_rec_std:.3f}")
        print(f"First Order Receiving Mean Score: {results.first_order_rec_mean:.3f}")
        print(f"First Order Receiving Std: {results.first_order_rec_std:.3f}")
        print(f"Second Order Receiving Mean Score: {results.second_order_rec_mean:.3f}")
        print(f"Second Order Receiving Std: {results.second_order_rec_std:.3f}")