#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

from pamogk.lib.sutils import *

sns.set()


def draw_heatmap(kernel, out_loc):
    """
    Parameters
    ----------
    kernel:
        list of kernel matrices
    out_loc:
        Output file location
    """
    plt.clf()
    sns.heatmap(kernel, vmin=0, vmax=1)
    plt.savefig(out_loc)


def calculate_kernel_density(kernel, threshold):
    """
    Parameters
    ----------
    kernel:
        one kernel matrix
    threshold:
        Looks for values greater than threshold

    Return
    -----------
    [0]:
        # of values greater than threshold
    [1]:
        Frequency of values greater than threshold (hit_values/all_values)
    """

    dim = len(kernel)
    total = 0
    count = 0
    for i in range(dim):
        for j in range(i):
            val = kernel[i][j]
            if val > threshold:
                total += kernel[i][j]
                count += 1
    total_count = dim * (dim - 1) / 2

    return count, float(total) / total_count


def calculate_kernel_avg(kernel):
    """
    Parameters
    ----------
    kernel:
        one kernel matrix
    Return
    -----------
    [0]:
        Average of values in kernel
    """
    dim = len(kernel)
    total = 0
    for i in range(dim):
        for j in range(i):
            total += kernel[i][j]
    total_count = dim * (dim - 1) / 2

    return float(total) / total_count


def calculate_kernel_variance(kernel):
    """
    Parameters
    ----------
    kernel:
        one kernel matrix
    Return
    -----------
    [0]:
        Variance of values in kernel
    """
    kernel_arr = kernel.flatten()
    total_sum = np.sum(kernel)
    n = len(kernel)
    mean = float(total_sum) / (n * n)
    for i in range(len(kernel_arr)):
        new_val = kernel_arr[i] - mean
        kernel_arr[i] = new_val * new_val

    return np.sum(kernel_arr) / (n * n)


'''
The below draw functions are helper functions to deal with multiple kernels or draw histograms
'''


# Saves heatmaps of kernels to out_folder location with name 0.png .... n.png
def draw_heatmaps(kernels, out_folder):
    for i, kernel in enumerate(kernels):
        draw_heatmap(kernel, out_folder / f'{i}.png')


def draw_special1_hist_for_kernels(kernels, bin_no, out_folder):
    """
    This function draws one histogram for each kernel in a set of kernels. Each histogram has bin_no bins.
     It saves histograms into out_folder location with name 0.png .... n.png
    kernels:
    bin_no:
    out_folder:
    """
    hist_arr = []
    for i, kernel in enumerate(kernels):
        val_arr = kernel.flatten()
        hist_arr.append(val_arr)

    bin_step = 1.0 / bin_no
    manual_bins = list(np.arange(0, 1.001, bin_step))
    for idx, kernel_values in enumerate(hist_arr):
        plt.clf()
        plt.hist(kernel_values, bins=manual_bins)
        plt.savefig(out_folder / f'{idx}.png')


def draw_hist_for_kernels(kernels, threshold, hist_type, out_file):
    """
    This function draws a variance, count or frequency histogram for each set of kernels.
     Variance does not care the threshold value.
    kernels:
    threshold:
    hist_type:
    out_file:
    """
    hist_arr = []
    for i, kernel in enumerate(kernels):
        if hist_type == 'variance':
            var = calculate_kernel_variance(kernel)
            hist_arr.append(var)
        elif hist_type == 'count':
            value, freq = calculate_kernel_density(kernel, threshold)
            hist_arr.append(value)
        elif hist_type == 'frequency':
            value, freq = calculate_kernel_density(kernel, threshold)
            hist_arr.append(freq)
    plt.clf()
    plt.hist(hist_arr, bins=50)
    plt.savefig(out_file)


def process_kernel_file(figure_type, process_type, in_file, extra):
    """
    Helper function to return the output_file name and kernels given that figure_type, process_type,
     kernel_file_loc and extra parameter if needed.
    figure_type:
    process_type:
    in_file:
    extra:
    """
    in_file = Path(in_file)
    folder = in_file.parent
    kernel_type = in_file.name[:-4]
    out_folder = folder / figure_type / process_type
    out_file = None
    if process_type == 'count' or process_type == 'frequency':
        out_file = out_folder / f'{kernel_type}-th={extra}.png'
    elif process_type == 'variance':
        out_file = out_folder / f'{kernel_type}.png'
    elif process_type == 'special1':
        out_file = out_folder / f'{kernel_type}-bin={extra}'
    elif figure_type == 'heatmaps':
        out_file = out_folder / kernel_type
    ensure_file_dir(out_file)
    kernels = np_load_data(in_file, key='kms')
    return out_file, kernels


# Rest is just helper functions to draw all figures in just one function.
def histogram_var(in_file):
    out_file, kernels = process_kernel_file('histograms', 'variance', in_file, 0)
    draw_hist_for_kernels(kernels, 0, 'variance', out_file)


def histogram_count(in_file, threshold):
    out_file, kernels = process_kernel_file('histograms', 'count', in_file, threshold)
    draw_hist_for_kernels(kernels, threshold, 'count', out_file)


def histogram_freq(in_file, threshold):
    out_file, kernels = process_kernel_file('histograms', 'frequency', in_file, threshold)
    draw_hist_for_kernels(kernels, threshold, 'frequency', out_file)


def histogram_special(in_file, bin_no):
    out_folder, kernels = process_kernel_file('histograms', 'special1', in_file, bin_no)
    safe_create_dir(out_folder)
    draw_special1_hist_for_kernels(kernels, bin_no, out_folder)


def heatmap_kernel(in_file):
    out_folder, kernels = process_kernel_file('heatmaps', '', in_file, 0)
    safe_create_dir(out_folder)
    draw_heatmaps(kernels, out_folder)


def main():
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alpha_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alpha_values = [0]
    kernel_types = ['rnaseq-kms', 'rppa-kms', 'som-kms']
    for alpha in alpha_values:
        for kernel in kernel_types:
            in_file = f'../data/pamogk_all/Experiment1-label=1-smoothing_alpha={alpha}-norm=True/{kernel}.npz'
            histogram_var(in_file)
            histogram_count(in_file, 0.2)
            histogram_freq(in_file, 0.2)
            histogram_special(in_file, 30)
            heatmap_kernel(in_file)


if __name__ == '__main__':
    main()
