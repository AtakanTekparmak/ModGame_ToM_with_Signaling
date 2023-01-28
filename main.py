from simulations.regular_simulation import RegularSimulation
from utilities import AgentsConfiguration

def main():
    agent_config: AgentsConfiguration = AgentsConfiguration(10, 5, 5)
    regular_simulation = RegularSimulation(agent_config=agent_config)
    regular_simulation.run(number_of_epochs=1000)
    regular_simulation.display_results()

if __name__ == '__main__':
    main()
