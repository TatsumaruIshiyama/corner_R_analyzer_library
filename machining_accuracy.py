#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from numpy import sin, cos, tan, arcsin, arccos, arctan, log, exp, radians, degrees, pi, sqrt
# %%
def R_mean(data, standard_column, standard_list):
    r_means = []
    for standard in standard_list:
        df = data.loc[data[standard_column] == standard].copy()
        if len(df) >= 1:
            df = df.reset_index(drop = True)
            df_common = df.loc[0, 'R':'z_5']
            df_common = pd.DataFrame(df_common).T
            df_common.loc[:, 'z_1':'z_5'] = df_common.loc[:, 'z_1':'z_5'] * -1
            r_mean = df.loc[:, 'r_1':'r_5'].mean()
            r_mean = pd.DataFrame(r_mean).T
            r_mean = pd.concat([df_common, r_mean], axis = 1)
            r_means.append(r_mean)
    r_mean = pd.concat(r_means)
    r_mean = r_mean.reset_index(drop = True)
    return r_mean
# %%
def plot_format(ax, xlim, ylim, fontsize = 7.5, flame_width = 1.5, scale_length = 5, pad = [0, 0], grid_width = 0.5):
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['font.family'] = 'Times New Roman'
    ax.spines["top"].set_linewidth(flame_width)
    ax.spines["left"].set_linewidth(flame_width)
    ax.spines["bottom"].set_linewidth(flame_width)
    ax.spines["right"].set_linewidth(flame_width)
    ax.minorticks_on()
    ax.tick_params(
        which = 'major',
        axis = 'y',
        direction = 'in',
        labelsize = fontsize,
        width = flame_width,
        length = scale_length,
        pad = pad[1]
        )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction = 'in',
        width = flame_width,
        length = scale_length * 0.7
        )
    ax.tick_params(
        which = 'major',
        axis = 'x',
        direction = 'in',
        labelsize = fontsize,
        width = flame_width,
        length = scale_length,
        pad = pad[0]
        )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction = 'in',
        width = flame_width,
        length = scale_length * 0.7
        )
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(color = 'black', linewidth = grid_width)
# %%
def plot_R_mean(ax, data, standard, label, size = 5, marker = ['o', 'o', 's', 'D', '^'], color = ['k', 'w', 'k', 'w', 'k']):
    for i in range(len(standard)):
        ax.plot(
            data.loc[i, 'r_1':'r_5'],
            data.loc[i, 'z_1':'z_5'] * -1,
            color = 'white',
            linewidth = 5
            )
    for i in range(len(standard)):
        ax.plot(
            data.loc[i, 'r_1':'r_5'],
            data.loc[i, 'z_1':'z_5'] * -1,
            color = 'black',
            linewidth = 1
            )
    for i in range(len(standard)):
        ax.plot(
            data.loc[i, 'r_1':'r_5'],
            data.loc[i, 'z_1':'z_5'] * -1,
            color = 'black',
            marker = marker[i],
            markersize = size,
            markerfacecolor = color[i],
            linewidth = 0,
            label = label[i]
            )
# %%
def S_left(r, r_prime):
    theta = arcsin(1 - r / r_prime)
    delta_r = r_prime - r
    S_r = pi * r ** 2 / 4
    S_r_prime = pi * r_prime ** 2 / 4

    S_all = S_r + 2 * r * delta_r + delta_r ** 2
    S_sector = r_prime ** 2 * theta / 2
    S_delta = r_prime * delta_r * (1 - sin(pi / 2 -theta) / 2) - S_sector

    S_left = S_all - S_r_prime - 2 * S_delta

    return S_left
# %%
def S_cut(r, ae):
    theta = arcsin(ae / r)
    phi = pi / 2 - 2 * theta
    S_r = r ** 2 * pi / 4

    S_tri = sqrt(2) * ae * r * sin(phi / 2) / 2
    S_sector = r ** 2 * phi / 2
    S_remove = S_sector - 2 * S_tri

    S_cut = S_r - S_remove
    return S_cut
# %%
def f_cross_section(r_list, z_list, delta_z = 0.01):
    r_list = np.array(r_list, dtype = 'float32')
    z_list = np.array(z_list, dtype = 'float32')
    n = len(r_list)
    d_r = np.diff(r_list)
    d_z = np.diff(z_list)
    a_list = d_r / d_z
    b_list = r_list[0:n -1] - a_list * z_list[0:n - 1]

    z_cross_section = np.arange(z_list[0], z_list[1], delta_z, dtype = 'float32')
    f_cross_section = a_list[0] * z_cross_section + b_list[0]
    for i in range(1, n - 1):
        z_i = np.arange(z_list[i], z_list[i + 1], delta_z, dtype = 'float32')
        f_i = (a_list[i] * z_i + b_list[i])
        f_cross_section = np.concatenate([f_cross_section, f_i])
        z_cross_section = np.concatenate([z_cross_section, z_i])
    return f_cross_section
# %%
def ratio_left_volume(r, data, d_z = 0.01, drop_negative = False):
    ratio_left = []
    drop_i = []
    df_ratio = data.copy()
    for i in range(len(data)):
        ae = data.loc[i, 'ae']
        r_list = data.loc[i, 'r_1':'r_5'].values
        z_list = data.loc[i, 'z_1':'z_5'].values
        z_calc = z_list[-1] - z_list[0]

        r_all = f_cross_section(r_list, z_list, delta_z = d_z)
        S = S_left(r, r_all)
        V_left = np.sum((S * d_z))
        V_cut = S_cut(r, ae) * z_calc
        ratio_left.append(100 * V_left / V_cut)
        if drop_negative:
            judge_drop = [j for j in S if j < 0]
            if judge_drop:
                drop_i.append(i)
    df_ratio['ratio_left'] = ratio_left
    df_ratio = df_ratio.drop(index = drop_i)
    return df_ratio
# %%
print('hello')