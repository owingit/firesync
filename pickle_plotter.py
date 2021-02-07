import simulation_plotter as sp
import simulation_helpers
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import csv

from scipy.stats import ks_2samp
from scipy.stats import kurtosis
from scipy.stats import skew


def pickle_plotter(data_path, other_path=None):
    # with open(data_path+'beta_sweep_individual.pickle', 'rb') as f_i:
    #     individual = pickle.load(f_i)
    # toggle
    individual = None
    with open(data_path+'beta_sweep_swarm.pickle', 'rb') as f_g:
        group = pickle.load(f_g)

    if other_path is not None:
        with open(data_path + 'beta_sweep_swarm_1.pickle', 'rb') as f_e:
            more_group_data = pickle.load(f_e)
        with open(data_path + 'beta_sweep_swarm_2.pickle', 'rb') as f_e2:
            extra_group_data = pickle.load(f_e2)
        for key in group.keys():
            for k in group[key].keys():
                for l in more_group_data[key][k]:
                    group[key][k].append(l)
                for l2 in extra_group_data[key][k]:
                    group[key][k].append(l2)

    ff_count = data_path.split('ff')[0][-2:]

    beta_dict = beta_plotter(individual=individual,
                             group=group,
                             data_path=data_path,
                             ff_count=ff_count)
    plot_ks_statistic(beta_dict, ff_count, data_path)


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

    # to_plot_individual_stats = {}
    # to_plot_individual_ps = {}
    # individual_beta_dict = beta_dict[0]
    # for beta in individual_beta_dict.keys():
    #     to_plot_individual_stats[beta], to_plot_individual_ps[beta] = (ks_2samp(individual_beta_dict[beta], raw_exp_y))
    #
    # xs = [b for b in to_plot_individual_stats.keys()]
    # y1s = [v for v in to_plot_individual_stats.values()]
    # y2s = [v for v in to_plot_individual_ps.values()]
    # ax.scatter(xs, y1s, label='Statistic: sim vs experiment')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('K-S Test p-value')
    # ax2.scatter(xs, y2s, label='P-value: sim vs experiment')
    #
    # plt.title('K-S Similarity vs Beta for {}ff (null: dists are same)'.format(ff_count))
    # plt.legend(loc='upper right')
    # plt.savefig(data_path + '_compare_distributions.png')
    # plt.cla()

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


def beta_plotter(individual, group, data_path, ff_count):
    independent_var = 'beta_{}ff'.format(ff_count)
    # dicts = [individual, group] #toggle
    dicts = [group]
    beta_dict = {}
    for i, d in enumerate(dicts):
        beta_dict[i] = {}
        for identifier, results in d.items():
            fig, ax = plt.subplots()
            ax.set_xlabel('Interburst interval [s]')
            ax.set_ylabel('Freq count')
            ax.set_xlim(left=0, right=120)
            exp_x, exp_y = plot_experimental_data(ax, data_path=data_path, ff_count=ff_count, plot=True)
            # the_x, the_y = plot_theoretical_data(ax, data_path=data_path, ff_count=ff_count, plot=True)
            beta = round(float(identifier.split('beta')[0].split('density')[1]), 4)
            _iids = []
            for k, iid_list in results.items():
                _iids.extend([x / 10 for iid in iid_list for x in iid])
            iids = [iid for iid in _iids if iid > 4]
            beta_dict[i][beta] = iids
            mean = round(float(np.mean(iids)), 4)
            std = round(float(np.std(iids)), 4)
            if ff_count == '01':
                bins = 50
            else:
                bins = 50

            # ccdfx, ccdfy = ccdf(np.array(iids))
            ys, bin_edges = np.histogram(iids, bins=bins, density=True)
            y_heights = [height for height in ys]
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            # ax.plot(ccdfx, ccdfy, color='blue', label='sim')
            ax.plot(bin_centers, y_heights, color='blue', label='sim')
            # ax.hist(iids, density=True, bins=bins, color='cyan', edgecolor='black', label='sim')
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
            plt.title('{}{}_{}mean_{}std'.format(
                beta,
                independent_var,
                mean,
                std
            ))
            plt.legend()
            plt.savefig('histograms/2_6/{}_interburst_dists_compared_w_experiments_{}{}.png'.format(
                string,
                beta,
                independent_var
            ))
            plt.clf()
            plt.close()

    return beta_dict


def plot_experimental_data(ax, data_path, ff_count, plot=False):
    x = []
    y = []
    with open(data_path+'ib{}ff.csv'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            if y_maybe > 3:
                y.append(y_maybe)
    if ff_count == '01':
        bins = 50
    else:
        bins = 50

    # ccdfx, ccdfy = ccdf(np.array(y))
    ys, bin_edges = np.histogram(y, bins=bins, density=True)
    y_heights = [height for height in ys]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if plot:
        # ax.plot(ccdfx, ccdfy, color='green', label='experimental')
        ax.plot(bin_centers, y_heights, color='green', label='experimental')
    return (bin_centers, ys), y


def plot_theoretical_data(ax, data_path, ff_count, plot=False):
    x = []
    y = []
    with open(data_path+'ib{}ffTheory.txt'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))
    if plot:
        ax.plot(x, y, color='red', label='theory')
    return x, y


def get_stats_from_experimental_data(data_path, ff_count):
    x = []
    y = []
    with open(data_path+'ib{}ff.csv'.format(ff_count), 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            x.append(i)
            y_maybe = float(row[0])
            if y_maybe > 3:
                y.append(y_maybe)
    mean = round(float(np.mean(y)), 4)
    std = round(float(np.std(y)), 4)
    skewness = skew(y)
    kurt = kurtosis(y)
    return mean, std, skewness, kurt, y


def pickle_beta_n_comparison(paths):
    comparison_dict = {}
    experimental_dict = {}
    for p in paths:
        data_path = 'data_from_cluster/raw_experiment_results/2_6/' + p
        ff_count = p.split('ff')[0]
        mean_exp, std_exp, skew_exp, kurtosis_exp, experimental_data = get_stats_from_experimental_data(
            data_path, ff_count)
        comparison_dict[ff_count] = {}
        experimental_dict[ff_count] = (mean_exp, std_exp, skew_exp, kurtosis_exp)
        with open(data_path + 'beta_sweep_swarm.pickle', 'rb') as f_g:
            data = pickle.load(f_g)
        with open(data_path + 'beta_sweep_swarm_1.pickle', 'rb') as f_g:
            extra_data = pickle.load(f_g)
        with open(data_path + 'beta_sweep_swarm_2.pickle', 'rb') as f_g2:
            extra_data_2 = pickle.load(f_g2)
        for key in data.keys():
            for k in data[key].keys():
                for l in extra_data[key][k]:
                    data[key][k].append(l)
                for l2 in extra_data_2[key][k]:
                    data[key][k].append(l2)

        for identifier, results in data.items():
            beta = round(float(identifier.split('beta')[0].split('density')[1]), 4)
            _iids = []
            for k, iid_list in results.items():
                _iids.extend([x / 10 for iid in iid_list for x in iid])
            iids = [iid for iid in _iids if iid > 3]
            mean = round(float(np.mean(iids)), 4)
            std = round(float(np.std(iids)), 4)
            skewness = skew(iids)
            kurt = kurtosis(iids)
            ks = ks_2samp(iids, experimental_data)
            comparison_dict[ff_count][beta] = (mean, std, skewness, kurt, ks)

    results_dict = {}
    for key in comparison_dict.keys():
        results_dict[key] = {'mean': [],
                             'std': [],
                             'skew': [],
                             'kurt': [],
                             'ks': []}
        comparator = experimental_dict[key]
        min_difference_means = float('inf')
        min_difference_stds = float('inf')
        min_difference_skews = float('inf')
        min_difference_kurts = float('inf')
        for beta, stats in comparison_dict[key].items():
            if abs(stats[0] - comparator[0]) < min_difference_means:
                results_dict[key]['mean'].append((beta, abs(stats[0] - comparator[0])))
                min_difference_means = abs(stats[0] - comparator[0])
            if abs(stats[1] - comparator[1]) < min_difference_stds:
                results_dict[key]['std'].append((beta, abs(stats[1] - comparator[1])))
                min_difference_stds = abs(stats[1] - comparator[1])
            if abs(stats[2] - comparator[2]) < min_difference_skews:
                results_dict[key]['skew'].append((beta, abs(stats[2] - comparator[2])))
                min_difference_stds = abs(stats[2] - comparator[2])
            if abs(stats[3] - comparator[3]) < min_difference_kurts:
                results_dict[key]['kurt'].append((beta, abs(stats[3] - comparator[3])))
                min_difference_stds = abs(stats[3] - comparator[3])
            results_dict[key]['ks'].append((beta, stats[4][0]))
    for key in results_dict:
        for sub_k in results_dict[key].keys():
            results_dict[key][sub_k].sort(key=lambda x: x[1])

    # use K-S right now
    to_plot = {}
    for key in results_dict:
        to_plot[key] = results_dict[key]['ks'][0][0]
    fig, ax = plt.subplots()
    ax.plot(list(to_plot.keys()), list(to_plot.values()), label='Best beta')
    ax.set_xlabel('N')
    ax.set_ylabel('Beta')
    plt.savefig('histograms/2_6/Beta_vs_N.png')


def main():
    data_path_preamble = 'data_from_cluster/raw_experiment_results/2_6/'
    extra_data_preamble = 'data_from_cluster/raw_experiment_results/2_3_moretrials/'
    paths = [
           # '01ff/',
             '05ff/',
             '10ff/',
             '15ff/',
             '20ff/']

    data_paths = [data_path_preamble + '{}'.format(i) for i in paths]
    other_paths = [extra_data_preamble + '{}'.format(i) for i in paths]
    pickle_beta_n_comparison(paths)
    for data_path, other_path in zip(data_paths, other_paths):
        pickle_plotter(data_path, other_path)


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