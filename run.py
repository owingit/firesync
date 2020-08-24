import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections


def main():
    thetastars = [math.pi / 8, math.pi / 4, 2 * math.pi]
    side_length = 20
    num_agents = 64
    step_count = 1000
    coupling_strength = 0.1
    num_trials = 1

    for trial in range(0, num_trials):
        for thetastar in [thetastars[-1]]:
            simulation = Simulation.Simulation(num_agents=num_agents,
                                               side_length=side_length,
                                               step_count=step_count,
                                               thetastar=thetastar,
                                               coupling_strength=coupling_strength,
                                               r_or_u="random")
            simulation.run()
            for firefly in simulation.agent_array:
                print(firefly.phase)
            simulation.animate_walk()


if __name__ == "__main__":
    main()
