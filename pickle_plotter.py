import csv
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu
from scipy.stats import kurtosis


def load_pickles(path):

    with open(path + 'beta_sweep_swarm.pickle', 'rb') as f_g:
        d = pickle.load(f_g)
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        try:
            with open(path + 'beta_sweep_swarm_{}.pickle'.format(i), 'rb') as f_g:
                extra_data = pickle.load(f_g)
                for key in d.keys():
                    for k in d[key].keys():
                        for l in extra_data[key][k]:
                            d[key][k].append(l)
        except FileNotFoundError:
            continue
    return d


def pickle_plotter(data_path, other_path=None):
    individual = None
    group = load_pickles(data_path)
    if other_path is not None:
        group_null = load_pickles(other_path)
    else:
        group_null = None

    ff_count = data_path.split('ff')[0][-2:]

    # compare_across_betas(group, ff_count)

    beta_dict = beta_plotter(individual=individual,
                             group=group,
                             data_path=data_path,
                             ff_count=ff_count,
                             group_null=group_null)
    # plot_ks_statistic(beta_dict, ff_count, data_path)


def compare_across_betas(data, num):
    fig, ax = plt.subplots()
    ax.set_xlabel('Interburst interval [s]')
    ax.set_ylabel('Probability density [1/s]')
    ax.set_xlim(left=0, right=120)
    for identifier, results in data.items():
        beta = round(float(identifier.split('beta')[0].split('density')[1]), 4)
        # if beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        _iids = []
        for k, iid_list in results.items():
            _iids.extend([x / 10 for iid in iid_list for x in iid])
        iids = [iid for iid in _iids if iid > 3]
        ys, bin_edges = np.histogram(iids, bins=np.arange(0.0, 1000, 3.0), density=True)
        y_heights = [height for height in ys]
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # ax.plot(ccdfx, ccdfy, color='blue', label='sim')
        # ax.hist(iids, bins=np.arange(0.0, 1000, 3.0), color='blue', density=True, label='sim', alpha=0.5)
        ax.plot(bin_centers, y_heights, label='{}'.format(beta), marker='2')
    ax.set_title('{}ff'.format(num))
    plt.legend()
    plt.savefig('data/betas_compared_{}ffsmalls.png'.format(num))


def plot_ks_statistic(beta_dict, ff_count, data_path):
    """Plot k-s test.

    Under the null hypothesis the two distributions are identical.
    If the K-S statistic is small or the p-value is high (greater than the significance level, say 5%),
    then we cannot reject the hypothesis that the distributions of the two samples are the same.
    Conversely, we can reject the null hypothesis if the p-value is low.
    """
    (exp_x, exp_y), raw_exp_y = plot_experimental_data(ax=None, data_path=data_path, ff_count=ff_count, plot=False)
    # the_x, the_y = plot_theoretical_data(ax=None, data_path=data_path, ff_count=ff_count, plot=False)
    fig, ax = plt.subplots()
    ax.set_xlabel('Beta')
    ax.set_ylabel('K-S Test Difference')
    to_plot_swarm_stats = {}
    to_plot_swarm_ps = {}
    swarm_beta_dict = beta_dict[0] # [1]

    for beta in swarm_beta_dict.keys():
        to_plot_swarm_stats[beta], to_plot_swarm_ps[beta] = (ks_2samp(swarm_beta_dict[beta], raw_exp_y))

    xs = [b for b in to_plot_swarm_stats.keys()]
    y1s = [v for v in to_plot_swarm_stats.values()]
    ax.scatter(xs, y1s, color='blue', label='Statistic: sim vs experiment')
    ax.set_xlabel('Beta')
    ax.set_ylabel('K-S Test Difference')

    plt.title('K-S Similarity vs Beta for {}ff'.format(ff_count))
    plt.legend()
    plt.savefig(data_path + '_compare_distributions_swarm.png')


def beta_plotter(individual, group, data_path, ff_count, group_null=None):
    independent_var = '{}ff'.format(ff_count)
    # dicts = [individual, group] #toggle
    dicts = [group]
    beta_dict = {}
    for i, d in enumerate(dicts):
        beta_dict[i] = {}
        for identifier, results in d.items():
            fig, ax = plt.subplots()
            ax.set_xlabel('Interburst interval [s]')
            ax.set_ylabel('Probability density [1/s]')
            ax.set_xlim(left=0, right=120)
            exp_x, exp_y = plot_experimental_data(ax, data_path=data_path, ff_count=ff_count, plot=True)
            the_x, the_y = plot_theoretical_data(ax, data_path=data_path, ff_count=ff_count, plot=True)
            beta = round(float(identifier.split('beta')[0].split('density')[1]), 4)
            _iids = []
            group_null_iids = []

            for k, iid_list in results.items():
                group_null_iid_list = group_null[identifier][k] if group_null is not None else []
                _iids.extend([x / 10 for iid in iid_list for x in iid])
                group_null_iids.extend([y / 10 for gn_iid in group_null_iid_list for y in gn_iid])
            iids = [iid for iid in _iids if iid > 3]
            group_null_iids = [gn_iid for gn_iid in group_null_iids if gn_iid > 3]
            beta_dict[i][beta] = iids
            mean = round(float(np.mean(iids)), 4)
            std = round(float(np.std(iids)), 4)

            # ccdfx, ccdfy = ccdf(np.array(iids))
            ys, bin_edges = np.histogram(iids, bins=np.arange(0.0, 1000, 3.0), density=True)
            y_heights = [height for height in ys]
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            # ax.plot(ccdfx, ccdfy, color='blue', label='sim')
            # ax.hist(iids, bins=np.arange(0.0, 1000, 3.0), color='blue', density=True, label='sim', alpha=0.5)
            if int(ff_count) == 5 or int(ff_count) == 20:
                if beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    with open('histograms/3_24/{}ff_data_sim_{}.csv'.format(ff_count, beta), 'w', newline='') as f:
                        d = {b: y for b, y in zip(bin_centers, y_heights)}
                        for key in d.keys():
                            f.write("%s,%s\n" % (key, d[key]))
            ax.plot(bin_centers, y_heights, color='black', lw='2', label='sim')
            if int(ff_count) == 1:
                ax.set_xlim(left=0, right=500)
            # ax.hist(iids, density=True, bins=bins, color='cyan', edgecolor='black', label='sim')
            if group_null is not None:
                gn_ys, gn_bin_edges = np.histogram(group_null_iids, bins=np.arange(0.0, 1000, 3.0),
                                                   density=True)
                gn_y_heights = [height for height in gn_ys]
                gn_bin_centers = 0.5 * (gn_bin_edges[1:] + gn_bin_edges[:-1])
                # ax.plot(ccdfx, ccdfy, color='blue', label='sim')
                ax.plot(gn_bin_centers, gn_y_heights, color='orange', label='sim_without_theory', marker='2')
            trials = len(iids)
            # toggle if i == 0:
            if i != 0:
                if trials > 1:
                    string = 'Individual_avg_over_' + str(trials) + '_{}mean_{}std'.format(mean, std)
                else:
                    string = 'Individual_avg' + '_{}mean_{}std'.format(mean, std)
                if '_obstacles' in identifier:
                    string = 'obs' + string
            else:
                if trials > 1:
                    string = 'Swarm_avg_over_' + str(trials) + '_{}mean_{}std'.format(mean, std)
                else:
                    string = 'Swarm_avg' + '_{}mean_{}std'.format(mean, std)
                if '_obstacles' in identifier:
                    string = 'obs' + string
            plt.title('{}'.format(independent_var))
            plt.legend()
            plt.savefig('histograms/3_24/{}_interburst_dists_{}_beta_from_mat{}.png'.format(
                string,
                beta,
                independent_var
            ))
            plt.clf()
            plt.close()

    return beta_dict


def order_parameter(iids):
    frequencies = [(math.pi * 2 * 1 / iid) for iid in iids]
    N = len(frequencies)
    r = 0
    for i in range(N):
        r += (math.e ** (frequencies[i]))
    print(frequencies)

    def circ_r(alpha, w=None, d=0, axis=0):
        if w is None:
            w = np.ones(alpha.shape)
        elif alpha.size is not w.size:
            raise ValueError("Input dimensions do not match")

        # Compute weighted sum of cos and sin of angles:
        r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

        # Obtain length:
        r = np.abs(r) / w.sum(axis=axis)

        # For data with known spacing, apply correction factor to
        # correct for bias in the estimation of r
        if d is not 0:
            c = d / 2 / np.sin(d / 2)
            r = c * r

        return np.array(r)

    return circ_r(np.array(frequencies))


def plot_experimental_data(ax, data_path, ff_count, plot=False):
    x = []
    y = []
    y_maybes = []
    with open(data_path+'ibs{}ff.csv'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            y_maybes.append(y_maybe)
            if y_maybe > 3:
                y.append(y_maybe)

    # ccdfx, ccdfy = ccdf(np.array(y))
    ys, bin_edges = np.histogram(y, bins=np.arange(0, 1000, 3.0), density=True)
    y_heights = [height for height in ys]

    y_heights.append(0.0)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    with open('histograms/3_24/{}ff_data_exp_dist.csv'.format(ff_count), 'w', newline='') as f:
        d = {b: y for b, y in zip(bin_centers, y_heights)}
        for key in d.keys():
            f.write("%s,%s\n" % (key, d[key]))
    if plot:
        # ax.plot(ccdfx, ccdfy, color='green', label='experimental')
        ax.hist(y, bins=np.arange(0, 1000, 3.0), color='cyan', edgecolor='black', label='experimental', density=True,
                alpha=0.8)
        # ax.plot(bin_edges, y_heights, color='green', label='experimental', marker='2')
    return (bin_centers, ys), y


def plot_theoretical_data(ax, data_path, ff_count, plot=False):
    x = []
    y = []
    with open(data_path+'ibs{}ffTheory.txt'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))
    if plot:
        ax.plot(x, y, color='red', label='theory', marker='*')
    return x, y


def get_stats_from_experimental_data(data_path, ff_count):
    x = []
    y = []
    all_ys = []
    with open(data_path+'ibs{}ff.csv'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            if y_maybe > 3:
                y.append(y_maybe)
    mean = round(float(np.mean(y)), 4)
    std = round(float(np.std(y)), 4)
    median = round(float(np.median(y)), 4)
    kurt = kurtosis(y)
    return mean, std, median, kurt, y


def pickle_beta_n_comparison(paths, extra=True):
    comparison_dict = {}
    experimental_dict = {}
    for p in paths:
        data_path = 'data_from_cluster/raw_experiment_results/' + p
        ff_count = p.split('ff')[0].split('/')[1]

        mean_exp, std_exp, med_exp, kurtosis_exp, experimental_data = get_stats_from_experimental_data(
            data_path, ff_count)
        comparison_dict[ff_count] = {}
        experimental_dict[ff_count] = (mean_exp, std_exp, med_exp, kurtosis_exp)
        with open(data_path + 'beta_sweep_swarm.pickle', 'rb') as f_g:
            data = pickle.load(f_g)
        if extra:
            for i in [1, 2, 3, 4, 5, 6, 7, 8]:
                try:
                    with open(data_path + 'beta_sweep_swarm_{}.pickle'.format(i), 'rb') as f_g:
                        extra_data = pickle.load(f_g)
                        for key in data.keys():
                            for k in data[key].keys():
                                for l in extra_data[key][k]:
                                    data[key][k].append(l)
                except FileNotFoundError:
                    continue

        for identifier, results in data.items():
            beta = round(float(identifier.split('beta')[0].split('density')[1]), 4)
            _iids = []
            for k, iid_list in results.items():
                _iids.extend([x / 10 for iid in iid_list for x in iid])

            iids = [iid for iid in _iids if iid > 3]
            mean = round(float(np.mean(iids)), 4)
            std = round(float(np.std(iids)), 4)
            med = round(float(np.median(iids)), 4)
            kurt = kurtosis(iids)

            iid_ys, bin_edges = np.histogram(iids, bins=np.arange(0.0, 1000, 3.0), density=True)
            # ax.plot(ccdfx, ccdfy, color='blue', label='sim')
            exp_ys, exp_bin_edges = np.histogram(experimental_data, bins=np.arange(0.0, 1000, 3.0),
                                                 density=True)
            ks = ks_2samp(iid_ys, exp_ys, mode='asymp')
            mw = mannwhitneyu(iids, experimental_data, alternative='two-sided')
            if not comparison_dict[ff_count].get(beta):
                comparison_dict[ff_count][beta] = (mean, std, med, kurt, ks, mw)

    results_dict = {}
    for key in comparison_dict.keys():
        results_dict[key] = {'mean': [],
                             'std': [],
                             'med': [],
                             'kurt': [],
                             'ks': [],
                             'ks_p': [],
                             'mannwhit': [],
                             'mannwhit_p': []}
        comparator = experimental_dict[key]
        min_difference_means = float('inf')
        min_difference_stds = float('inf')
        min_difference_meds = float('inf')
        min_difference_kurts = float('inf')
        for beta, stats in comparison_dict[key].items():
            results_dict[key]['mean'].append((beta, abs(stats[0] - comparator[0])))
            if abs(stats[1] - comparator[1]) < min_difference_stds:
                results_dict[key]['std'].append((beta, abs(stats[1] - comparator[1])))
                min_difference_stds = abs(stats[1] - comparator[1])
            if abs(stats[2] - comparator[2]) < min_difference_meds:
                results_dict[key]['med'].append((beta, abs(stats[2] - comparator[2])))
                min_difference_stds = abs(stats[2] - comparator[2])
            if abs(stats[3] - comparator[3]) < min_difference_kurts:
                results_dict[key]['kurt'].append((beta, abs(stats[3] - comparator[3])))
                min_difference_stds = abs(stats[3] - comparator[3])
            # if stats[4][1] > 0.05:
            results_dict[key]['ks'].append((beta, stats[4][0]))
            results_dict[key]['ks_p'].append((beta, stats[4][1]))
            results_dict[key]['mannwhit'].append((beta, stats[5][0]))
            results_dict[key]['mannwhit_p'].append((beta, stats[5][1]))
    for key in results_dict:
        for sub_k in results_dict[key].keys():
            results_dict[key][sub_k].sort(key=lambda x: x[0])

    ys = np.zeros((50, 4))

    for i, key in enumerate(results_dict):
        if key != '01':
            for e, pair in enumerate(results_dict[key]['med']):
                ys[e, i-1] = pair[1]
    fig2, ax2 = plt.subplots()
    xticks = ['05', '10', '15', '20']
    yticks = np.arange(0.0, 1.0, 0.02)
    ax2 = sns.heatmap(ys[::2], linewidth=0.0, xticklabels=xticks,
                      yticklabels=yticks[::2],
                      cmap=sns.cm.rocket_r)

    ax2.set_xticklabels(xticks)
    ax2.set_yticklabels(yticks[::2])
    ax2.invert_yaxis()
    plt.xlabel('N')
    plt.ylabel('Beta')
    plt.title('Mean difference from experimental data, β vs N')
    plt.savefig('histograms/3_24/Beta_vs_N_heatmap.png')


def main():
    data_path_preamble = 'data_from_cluster/raw_experiment_results/'
    paths = [
             '3_24/01ff/',
             '3_24/05ff/',
             '3_24/10ff/',
             '3_24/15ff/',
             '3_24/20ff/'
            ]
    null_paths = paths # temporary

    data_paths = [data_path_preamble + '{}'.format(i) for i in paths]
    null_paths = [data_path_preamble + '{}'.format(i) for i in null_paths]
    pickle_beta_n_comparison(paths, extra=True)
    for data_path, other_path in zip(data_paths, null_paths):
        pickle_plotter(data_path, other_path=None)


def cdf(a):
    X = np.sort(np.array(a))
    Y = np.array(range(a.size)) / float(a.size)
    return X, Y


def ccdf(a):
    X = np.sort(np.array(a))
    Y = (np.full(a.shape, fill_value=a.size) - np.array(range(a.size))) / float(a.size)
    return X, Y


if __name__ == '__main__':
    main()
