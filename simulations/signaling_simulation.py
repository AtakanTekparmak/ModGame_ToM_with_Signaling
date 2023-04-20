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
from agents.agent import SignalingAgent
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
    an AgentsConfiguration instance.

    Args:
        agent_config: AgentsConfiguration == The Agent Population Config
    '''
    # Extract agent numbers
    number_0 = agent_config.zero_order_sig_agent_number
    number_1 = agent_config.zero_order_agent_number
    number_2 = agent_config.first_order_sig_agent_number
    number_3 = agent_config.first_order_agent_number
    number_4 = agent_config.second_order_sig_agent_number
    number_5 = agent_config.second_order_agent_number

    # Create a list of zero-order agents
    zero_order_sig_agents = [ZeroOrderSignalingAgent() for _ in range(number_0)]
    zero_order_agents = zero_order_sig_agents + [ZeroOrderTheoryOfMindAgent() for _ in range(number_1)]

    # Create a list of first-order agents, each with a reference to a zero-order agent
    first_order_sig_agents = [FirstOrderSignalingAgent() for _ in range(number_2)]
    first_order_agents = first_order_sig_agents + [FirstOrderTheoryOfMindAgent() for _ in range(number_3)]

    # Create a list of second-order agents, each with a reference to a zero and first-order agent
    second_order_sig_agents = [SecondOrderSignalingAgent() for _ in range(number_4)]
    second_order_agents = second_order_sig_agents + [SecondOrderTheoryOfMindAgent() for _ in range(number_5)]

    # Create a list of all agents
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
        #self.receiving_agents: 
        self.signaling_agent_actions = [[] for _ in self.signaling_agents]
        self.signaling_agent_scores = [0 for _ in self.signaling_agents]

    def simulate_round(self) -> None:
        # Have each agent decide on an action_
        signals = [agent.signal() for agent in self.signaling_agents if isinstance(agent, ZeroOrderSignalingAgent)]
        actions = [agent.decide() for agent in self.signaling_agents]
        actions_copy = actions

        for index, action in enumerate(actions):
            lower_choices = sum([1 for choice in actions_copy if choice == (action - 1) % 23])
            self.signaling_agent_scores[index] += lower_choices

        # Update the beliefs of each agent based on the actions of all agents
        for agent, action in zip(self.signaling_agents, actions):
            agent.update(action)

    def get_results(self) -> RegularSimulationResults:
        '''
        Calculates the statistics and returns 
        them in a dict with individual results.
        '''
        # Extract agent numbers
        number_0 = self.agent_config.zero_order_sig_agent_number
        number_1 = self.agent_config.zero_order_agent_number
        number_2 = self.agent_config.first_order_sig_agent_number
        number_3 = self.agent_config.first_order_agent_number
        number_4 = self.agent_config.second_order_sig_agent_number

        # Slice scores per ToM level
        zero_order_sig_scores = self.signaling_agent_scores[:number_0]
        zero_order_scores = self.signaling_agent_scores[number_0:number_1]
        first_order_sig_scores = self.signaling_agent_scores[number_1:number_2]
        first_order_scores = self.signaling_agent_scores[number_2:number_3]
        second_order_sig_scores = self.signaling_agent_scores[number_3:number_4]
        second_order_scores = self.signaling_agent_scores[number_4:]


        return SignalingSimulationResults(
            zero_order_sig_mean=get_mean(zero_order_sig_scores),
            zero_order_sig_std=get_std(zero_order_sig_scores),
            zero_order_mean=get_mean(zero_order_scores),
            zero_order_std=get_std(zero_order_scores),
            first_order_sig_mean=get_mean(first_order_sig_scores),
            first_order_sig_std=get_std(first_order_sig_scores),
            first_order_mean=get_mean(first_order_scores),
            first_order_std=get_std(first_order_scores),
            second_order_sig_mean=get_mean(second_order_sig_scores),
            second_order_sig_std=get_std(second_order_sig_scores),
            second_order_mean=get_mean(second_order_scores),
            second_order_std=get_std(second_order_scores),
        )

    def display_results(self, results: SignalingSimulationResults) -> None:

        print("Printing statistics ...")
        print(f"Zero Order Signaling Mean Score: {results.zero_order_sig_mean:.3f}")
        print(f"Zero Order Signaling Std: {results.zero_order_sig_std:.3f}")
        print(f"Zero Order Mean Score: {results.zero_order_mean:.3f}")
        print(f"Zero Order Std: {results.zero_order_std:.3f}")
        print(f"First Order Signaling Mean Score: {results.first_order_sig_mean:.3f}")
        print(f"First Order Signaling Std: {results.first_order_sig_std:.3f}")
        print(f"First Order Mean Score: {results.first_order_mean:.3f}")
        print(f"First Order Std: {results.first_order_std:.3f}")
        print(f"Second Order Signaling Mean Score: {results.second_order_sig_mean:.3f}")
        print(f"Second Order Signaling Std: {results.second_order_sig_std:.3f}")
        print(f"Second Order Mean Score: {results.second_order_mean:.3f}")
        print(f"Second Order Std: {results.second_order_std:.3f}")