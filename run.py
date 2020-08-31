import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections
from datetime import datetime


def main():
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    side_length = 16
    num_agent_options = [64]  #, 500, 1000]
    step_count = 200
    coupling_strengths = [0.05]  #, 0.2, 0.5]
    num_trials = 1

    simulations = []
    experiment_results = {}
    now = datetime.now()
    for thetastar in thetastars:
        for num_agents in num_agent_options:
            for coupling_strength in coupling_strengths:
                for Tb in inter_burst_intervals:
                    for trial in range(0, num_trials):
                        simulation = Simulation.Simulation(num_agents=num_agents,
                                                           side_length=side_length,
                                                           step_count=step_count,
                                                           thetastar=thetastar,
                                                           coupling_strength=coupling_strength,
                                                           Tb=Tb,
                                                           r_or_u="random",
                                                           use_obstacles=True)
                        simulations.append(simulation)

    for simulation in simulations:
        simulation.run()
        result_key = frozenset((simulation.tstar_seed,
                                simulation.Tb,
                                simulation.total_agents,
                                simulation.coupling_strength))
        if experiment_results.get(result_key):
            experiment_results[result_key].append(simulation)
        else:
            experiment_results[result_key] = [simulation]
        simulation.animate_phase_bins(now, show_gif=True, write_gif=False)
        simulation.animate_walk(now, show_gif=True, write_gif=False)
    print(experiment_results)


if __name__ == "__main__":
    main()
