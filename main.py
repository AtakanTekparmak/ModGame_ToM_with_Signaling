from simulations.regular_simulation import RegularSimulation
from utilities import AgentsConfiguration, RegularSimulationResults, get_mean
from typing import List

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

def main():
    # Define population configuration
    agent_config: AgentsConfiguration = AgentsConfiguration(120, 90, 90)
    results: List[RegularSimulationResults] = []
    epochs: int = 1000

    # Run it 10 times to aggregate results
    for _ in range(10):
        regular_simulation = RegularSimulation(agent_config=agent_config)
        regular_simulation.run(number_of_epochs=epochs)
        results.append(regular_simulation.get_results())

    # Display aggregate results
    regular_simulation.display_results(results=aggregate_results(results, epochs))


if __name__ == '__main__':
    main()
