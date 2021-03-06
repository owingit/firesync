import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


class TSA:
    def __init__(self, experiment_results, now):
        self.experiment_results = experiment_results
        self.now = now

    def fourier_transform(self):
        thresh = 0
        ffts = {}
        for identifier, simulation_list in self.experiment_results.items():
            try:
                beta = round(float(identifier.split('beta')[0].split('density')[1]), 3)
            except AttributeError:
                beta = list(identifier)[2]
            ffts[beta] = {}
            for i, simulation in enumerate(simulation_list):
                ff_count = len(simulation.firefly_array)
                all_flashes = []
                for ff in simulation.firefly_array:
                    all_flashes.extend([i for i, val in enumerate(ff.flashed_at_this_step[thresh:]) if val])
                all_flashes = sorted(all_flashes)
                signal = np.zeros(max(all_flashes))
                for index in all_flashes:
                    signal[index-1] += 1
                # Number of samplepoints
                N = len(signal)
                # sample spacing
                yf = fft(signal)
                ffts[beta][i] = yf
                xf = np.fft.fftfreq(N, 0.1)
                fftData = np.fft.fftshift(yf)
                freq = np.fft.fftshift(xf)
                fig, ax = plt.subplots()
                ax.plot(freq, 2.0 / N * np.abs(fftData[:N]))
                ax.set_xlim((-2, 2))
                ax.set_xlabel('f (Hz)')
                ax.set_ylabel('|P(f)|')
                plt.savefig('data/{}ff_{}beta_fourier_{}.png'.format(ff_count, beta, i))

    def width_histogram(self):
        results = {}
        cutoff_results = {}
        peak_results = {}
        s = None
        for identifier, simulation_list in self.experiment_results.items():
            beta = list(identifier)[0]
            results[beta] = {}
            cutoff_results[beta] = {}
            peak_results[beta] = {}
            for i, simulation in enumerate(simulation_list):
                s = simulation
                _, peak_results[beta][i], cutoff_results[beta][i], results[beta][i] = simulation.peak_variances(
                    thresh=0
                )

        sorted_results = {k: results[k] for k in sorted(results)}
        all_widths = {}
        for beta in sorted_results.keys():
            all_widths[beta] = []
            for instance in sorted_results[beta].keys():
                for _, y1 in sorted_results[beta][instance].items():
                    all_widths[beta].append(y1)

        sorted_peak_results = {k: peak_results[k] for k in sorted(peak_results)}
        all_peaks = {}
        for beta in sorted_peak_results.keys():
            all_peaks[beta] = []
            for instance in sorted_peak_results[beta].keys():
                all_peaks[beta].extend(sorted_peak_results[beta][instance])
        self.plot_height_histograms(all_peaks, len(s.firefly_array))
        self.plot_width_histograms(all_widths, len(s.firefly_array))

    @staticmethod
    def plot_height_histograms(results, ff_count):
        fig, ax = plt.subplots()

        for beta in results.keys():
            ys, edges = np.histogram(results[beta], density=True)
            ax.plot(edges[:-1], ys, label='{}'.format(beta), marker='2')
        ax.set_xlabel('Height (#)')
        ax.set_ylabel('Probability density')
        ax.set_title('Height histogram, {}ff'.format(ff_count))
        plt.legend()
        plt.savefig('data/height_histogram_{}ff'.format(ff_count))

    @staticmethod
    def plot_width_histograms(results, ff_count):
        fig, ax = plt.subplots()

        for beta in results.keys():
            ys, edges = np.histogram(results[beta], density=True)
            ax.plot(edges[:-1], ys, label='{}'.format(beta), marker='2')
        ax.set_xlabel('Width (s)')
        ax.set_ylabel('Probability density')
        ax.set_title('Width histogram, {}ff'.format(ff_count))
        plt.legend()
        plt.savefig('data/width_histogram_{}ff'.format(ff_count))

    def plot_widths(self):
        steps = 50000  # num simulation steps TODO: extract this
        results = {}
        peak_results = {}
        cutoff_results = {}
        for identifier, simulation_list in self.experiment_results.items():
            beta = list(identifier)[0]
            peak_results[beta] = {}
            results[beta] = {}
            cutoff_results[beta] = {}
            for i, simulation in enumerate(simulation_list):
                peak_results[beta][i], _, cutoff_results[beta][i], results[beta][i] = simulation.peak_variances(thresh=0)
        all_scatterpoints = {}
        print(peak_results)
        sorted_peak_results = {k: peak_results[k] for k in sorted(peak_results)}
        all_peaks = {}
        for beta in sorted_peak_results.keys():
            all_peaks[beta] = []
            for instance in sorted_peak_results[beta].keys():
                all_peaks[beta].extend(sorted_peak_results[beta][instance])
        fig, ax = plt.subplots()
        sorted_results = {k: results[k] for k in sorted(results)}
        for beta in sorted_results.keys():
            all_scatterpoints[beta] = []
            burst_avgs = []
            for instance in sorted_results[beta].keys():
                timescale = steps / len(sorted_results[beta][instance])
                scatterpoints = {}
                for x1, y1 in sorted_results[beta][instance].items():
                    if not scatterpoints.get(x1):
                        scatterpoints[x1] = [y1]
                    else:
                        scatterpoints[x1].append(y1)
                for s in scatterpoints.keys():
                    all_vals_at_burst = scatterpoints[s]
                    avg_at_burst = sum(all_vals_at_burst) / len(all_vals_at_burst)
                    burst_avgs.append((s, avg_at_burst))
            all_scatterpoints[beta].extend(burst_avgs)

            x = [tup[0] * timescale for tup in all_scatterpoints[beta]]
            y = [tup[1] for tup in all_scatterpoints[beta]]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", label='beta={}'.format(beta))
            ax.set_ylim([0, 500])

        ax.set_xlabel('T')
        ax.set_ylabel('Width variance')
        ax.set_title('{}ff'.format(len(simulation.firefly_array)))
        plt.legend()
        plt.savefig('data/{}ff_width_variances.png'.format(len(simulation.firefly_array)))

    def plot_interburst_intervals(self):
        results = {}
        steps = 50000 # num simulation steps TODO: extract this
        for identifier, simulation_list in self.experiment_results.items():
            try:
                beta = round(float(identifier.split('beta')[0].split('density')[1]), 3)
            except AttributeError:
                beta = list(identifier)[0]
            results[beta] = {}
            for i, simulation in enumerate(simulation_list):
                results[beta][i] = simulation.temporal_interburst_dist(thresh=0)
        all_scatterpoints = {}

        fig, ax = plt.subplots()
        sorted_results = {k: results[k] for k in sorted(results)}
        for beta in list(sorted_results.keys()):
            all_scatterpoints[beta] = []
            for instance in sorted_results[beta].keys():
                scatterpoints = [(x1, y1) for x1, y1 in sorted_results[beta][instance].items() if y1 > 0]
                timescale = steps / len(sorted_results[beta][instance])
                all_scatterpoints[beta].extend(scatterpoints)

            x = [tup[0] * timescale for tup in all_scatterpoints[beta]]
            y = [tup[1] for tup in all_scatterpoints[beta]]

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", label='beta={}'.format(beta))
            ax.set_xlim([0, steps])
            ax.set_ylim([0, 500])

        ax.set_xlabel('T')
        ax.set_ylabel('Interburst-interval distance')
        ax.set_title('{}ff'.format(len(simulation.firefly_array)))
        plt.legend()
        plt.savefig('data/{}ff_interburst_interval_distances.png'.format(len(simulation.firefly_array)))
