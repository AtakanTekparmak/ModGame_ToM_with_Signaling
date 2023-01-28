from typing import List

from agents.regular_agents import (
    TheoryOfMindAgent, 
    ZeroOrderTheoryOfMindAgent, 
    FirstOrderTheoryOfMindAgent, 
    SecondOrderTheoryOfMindAgent
)
from utilities import AgentsConfiguration
from simulations.simulation import Simulation
        
def create_agents(agent_config: AgentsConfiguration = AgentsConfiguration(10, 10, 10)) -> List[TheoryOfMindAgent]:
    ''' 
    Creates a set population of agents based on 
    an AgentsConfiguration instance.

    Args:
        agent_config: AgentsConfiguration == The Agent Population Config
    '''
    # Extract agent numbers
    number_0 = agent_config.zero_order_agent_number
    number_1 = agent_config.first_order_agent_number
    number_2 = agent_config.second_order_agent_number

    # Create a list of zero-order agents
    zero_order_agents = [ZeroOrderTheoryOfMindAgent() for _ in range(number_0)]

    # Create a list of first-order agents, each with a reference to a zero-order agent
    first_order_agents = [FirstOrderTheoryOfMindAgent() for _ in range(number_1)]

    # Create a list of second-order agents, each with a reference to a zero and first-order agent
    second_order_agents = [SecondOrderTheoryOfMindAgent() for _ in range(number_2)]

    # Create a list of all agents
    return zero_order_agents + first_order_agents + second_order_agents

def agent_decide_and_save_data(agent_actions: List[List[int]], agent: TheoryOfMindAgent, index: int) -> int:
    '''
    Gets the agent decision, updates the 
    agent actions and returns the decision.
    '''
    # Get agent decision
    agent_decision: int = agent.decide()
    agent_actions[index].append(agent_decision)
    return agent_decision

class RegularSimulation(Simulation):
    '''
    A Simulation of Theory of Mind
    agents playing the mod game with
    23 choices, without signaling.
    '''
    def __init__(
        self,
        agent_config: AgentsConfiguration = AgentsConfiguration(5,5,5)
    ):
        super().__init__(agent_config)
        self.agents: List[TheoryOfMindAgent] = create_agents(agent_config)
        self.agent_actions = [[] for _ in self.agents]
        self.agent_scores = [0 for _ in self.agents]

    def simulate_round(self) -> None:
        # Have each agent decide on an action
        actions = [agent_decide_and_save_data(self.agent_actions, agent, index) for index, agent in enumerate(self.agents)]
        actions_copy = actions

        for index, action in enumerate(actions):
            lower_choices = sum([1 for choice in actions_copy if choice == (action - 1) % 23])
            self.agent_scores[index] += lower_choices

        # Update the beliefs of each agent based on the actions of all agents
        for agent, action in zip(self.agents, actions):
            agent.update(action)

    def display_results(self) -> None:
        # Extract agent numbers
        number_0 = self.agent_config.zero_order_agent_number
        number_1 = self.agent_config.first_order_agent_number
        number_2 = self.agent_config.second_order_agent_number

        # Slice scores per ToM level
        zero_order_scores = self.agent_scores[:number_0]
        first_order_scores = self.agent_scores[number_0:number_0+number_1]
        second_order_scores = self.agent_scores[number_0+number_1:]

        print("Printing scores...")
        print(f"Zero Order Agents ({number_0}): ")
        print(zero_order_scores)
        print(f"First Order Agents ({number_1}): ")
        print(first_order_scores)
        print(f"Second Order Agents ({number_2}): ")
        print(second_order_scores)


