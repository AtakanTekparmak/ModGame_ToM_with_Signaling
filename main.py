from simulations.regular_simulation import RegularSimulation
from simulations.signaling_simulation import SignalingSimulation
from utilities import (
    AgentsConfiguration, 
    SignalingAgentsConfiguration,
    RegularSimulationResults, 
    SignalingSimulationResults,
    get_mean
)
from typing import List
from agents.signaling_agents import ZeroOrderSignalingAgent

def aggregate_results(
        results: List[RegularSimulationResults],
        epochs: int
    ) -> RegularSimulationResults:
    '''
    Returns the aggregated mean of a list
    of RegularSimulationResults (all statistics
    provided are per round).
    '''
    return RegularSimulationResults (
        zero_order_mean=get_mean([result.zero_order_mean for result in results]) / epochs,
        zero_order_std=get_mean([result.zero_order_std for result in results]) / epochs,
        first_order_mean=get_mean([result.first_order_mean for result in results]) / epochs,
        first_order_std=get_mean([result.first_order_std for result in results]) / epochs,
        second_order_mean=get_mean([result.second_order_mean for result in results]) / epochs,
        second_order_std=get_mean([result.second_order_std for result in results]) / epochs,
    )

def aggregate_signaling_results(
        results: List[SignalingSimulationResults],
        epochs: int
    ) -> SignalingSimulationResults:
    '''
    Returns the aggregated mean of a list
    of SignalingSimulationResults (all statistics
    provided are per round).
    '''
    return SignalingSimulationResults (
        zero_order_sig_mean=get_mean([result.zero_order_sig_mean for result in results]) / epochs,
        zero_order_sig_std=get_mean([result.zero_order_sig_std for result in results]) / epochs,
        zero_order_mean=get_mean([result.zero_order_mean for result in results]) / epochs,
        zero_order_std=get_mean([result.zero_order_std for result in results]) / epochs,
        first_order_sig_mean=get_mean([result.first_order_sig_mean for result in results]) / epochs,
        first_order_sig_std=get_mean([result.first_order_sig_std for result in results]) / epochs,
        first_order_mean=get_mean([result.first_order_mean for result in results]) / epochs,
        first_order_std=get_mean([result.first_order_std for result in results]) / epochs,
        second_order_sig_mean=get_mean([result.second_order_sig_mean for result in results]) / epochs,
        second_order_sig_std=get_mean([result.second_order_sig_std for result in results]) / epochs,
        second_order_mean=get_mean([result.second_order_mean for result in results]) / epochs,
        second_order_std=get_mean([result.second_order_std for result in results]) / epochs,
    )

def run_regular_simulation(zero_order_number: int = 100, first_order_number: int = 100, second_order_number: int = 100):
    ''' Runs a regular simulation 10 times and aggregates the results.'''
    # Define population configuration
    agent_config: AgentsConfiguration = AgentsConfiguration(zero_order_number, first_order_number, second_order_number)
    results: List[RegularSimulationResults] = []
    epochs: int = 1000

    # Run it 10 times to aggregate results
    for _ in range(10):
        regular_simulation = RegularSimulation(agent_config=agent_config)
        regular_simulation.run(number_of_epochs=epochs)
        results.append(regular_simulation.get_results())

    # Display aggregate results
    regular_simulation.display_results(results=aggregate_results(results, epochs))

def run_signaling_simulation():
    ''' Runs a signaling simulation 10 times and aggregates the results.'''
    # Define population configuration
    agent_config:  SignalingAgentsConfiguration = SignalingAgentsConfiguration(50, 50, 50, 50, 50, 50)
    results: List[SignalingSimulationResults] = []
    epochs: int = 1000

    # Run it 10 times to aggregate results
    for _ in range(10):
        signaling_simulation = SignalingSimulation(agent_config=agent_config)
        signaling_simulation.run(number_of_epochs=epochs)
        results.append(signaling_simulation.get_results())

    # Display aggregate results
    signaling_simulation.display_results(results=aggregate_signaling_results(results, epochs))

def main():
    run_regular_simulation(100, 100, 1200)
    print('-----------------')
    run_signaling_simulation()

if __name__ == '__main__':
    main()
