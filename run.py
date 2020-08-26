import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections
from datetime import datetime


def main():
    thetastars = [math.pi / 8, math.pi / 4, 2 * math.pi]
    side_length = 10
    num_agents = 100
    step_count = 1000
    coupling_strength = 0.5
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
            now = datetime.now()
            simulation.animate_phase_bins(now, show_gif=False, write_gif=True)
            simulation.animate_walk(now, show_gif=False, write_gif=True)


if __name__ == "__main__":
    main()
