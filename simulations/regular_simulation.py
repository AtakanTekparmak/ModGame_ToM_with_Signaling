from typing import List

from agents.regular_agents import (
    ZeroOrderTheoryOfMindAgent, 
    FirstOrderTheoryOfMindAgent, 
    SecondOrderTheoryOfMindAgent
)
from agents.agent import TheoryOfMindAgent
from utilities import AgentsConfiguration, RegularSimulationResults, get_mean, get_std
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

    def add_agents(self, agents: List[TheoryOfMindAgent]) -> None:
        '''
        Adds a list of agents to the simulation.
        '''
        self.agents += agents
        self.agent_actions += [[] for _ in agents]
        self.agent_scores += [0 for _ in agents]

    def get_results(self) -> RegularSimulationResults:
        '''
        Calculates the statistics and returns 
        them in a dict with individual results.
        '''
        # Extract agent numbers
        number_0 = self.agent_config.zero_order_agent_number
        number_1 = self.agent_config.first_order_agent_number

        # Slice scores per ToM level
        zero_order_scores = self.agent_scores[:number_0]
        first_order_scores = self.agent_scores[number_0:number_0+number_1]
        second_order_scores = self.agent_scores[number_0+number_1:]

        return RegularSimulationResults(
            zero_order_mean=get_mean(zero_order_scores),
            zero_order_std=get_std(zero_order_scores), 
            first_order_mean=get_mean(first_order_scores),
            first_order_std=get_std(first_order_scores),
            second_order_mean=get_mean(second_order_scores),
            second_order_std=get_std(second_order_scores),
        )



    def display_results(
        self, 
        results: RegularSimulationResults, 
        print_individual_scores: bool = False
    ) -> None:

        print("Printing statistics ...")
        print(f"Zero Order Mean Score: {results.zero_order_mean:.3f}")
        print(f"Zero Order Std: {results.zero_order_std:.3f}")
        print(f"First Order Mean Score: {results.first_order_mean:.3f}")
        print(f"First Order Std: {results.first_order_std:.3f}")
        print(f"Second Order Mean Score: {results.second_order_mean:.3f}")
        print(f"Second Order Std: {results.second_order_std:.3f}")

        if not print_individual_scores:
            return

        print("----------------------------")
        print("Printing scores...")
        print(f"Zero Order Agents: ")
        print(results.zero_order_results)
        print(f"First Order Agents: ")
        print(results.first_order_results)
        print(f"Second Order Agents: ")
        print(results.second_order_results)


