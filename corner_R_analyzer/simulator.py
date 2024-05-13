# %%
import math
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
import glob
import os
from numpy import sin, cos, tan, arcsin, arccos, arctan, log, exp, radians, degrees, pi, sqrt
import corner_R_analyzer.machining_accuracy as machining_accuracy
import corner_R_analyzer.calcurate as calc
import time
import tqdm
import IPython
from IPython.display import clear_output
import moviepy.editor as mp
# %%
def plot_format(ax, xlim, ylim, fontsize = 7.5, flame_width = 1.5, scale_length = 5, pad = [0, 0], grid_width = 0.5, grid_color = 'k', grid_which = 'major', minor_grid_axis = 'both'):
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
    ax.grid(color = grid_color, linewidth = grid_width, which = 'major')
    if grid_which == 'both':
        ax.grid(
            color = grid_color,
            linewidth = grid_width * 0.6,
            which = 'minor',
            axis = minor_grid_axis
        )
# %%
def arc_coordinate(R, D, ae, N = 100, type = ''):
    r = D / 2
    x_lim = -R * 3
    y_lim = -R * 3

    t_work = np.linspace(0, pi / 2, N)
    x_work_a = np.zeros(3 * N)
    y_work_a = np.zeros(3 * N)
    x_work_b = np.zeros(3 * N)
    y_work_b = np.zeros(3 * N)
    x_path_a = np.zeros(3 * N)
    y_path_a = np.zeros(3 * N)
    x_path_b = np.zeros(3 * N)
    y_path_b = np.zeros(3 * N)

    x_work_a[N:2 * N] = R * cos(t_work)
    x_work_a[:N] = [x_work_a[N]] * N
    x_work_a[2 * N:] = np.linspace(x_work_a[2 * N - 1], x_lim, N)
    y_work_a[N:2 * N] = R * sin(t_work)
    y_work_a[:N] = np.linspace(y_lim, y_work_a[N], N)
    y_work_a[2 * N:] = [y_work_a[2 * N - 1]] * N

    if type == 'concentric' or R - ae >= r:
        t_work = np.linspace(0, pi / 2, N)
        x_work_b[N:2 * N] = (R - ae) * cos(t_work)
        y_work_b[N:2 * N] = (R - ae) * sin(t_work)
        x_work_b[:N] = [x_work_b[N]] * N
        y_work_b[:N] = np.linspace(y_lim, y_work_b[N], N)
        x_work_b[2 * N:] = np.linspace(x_work_b[2 * N - 1], x_lim, N)
        y_work_b[2 * N:] = [y_work_b[2 * N - 1]] * N

        x_path_b[N:2 * N] = (R - r - ae) * cos(t_work)
        y_path_b[N:2 * N] = (R - r - ae) * sin(t_work)

    if type == 'inner' or R - ae < r:
        p = y_work_a[1] - y_work_a[0]
        N_ae = int(ae / p)
        t_work_prime = np.linspace(0, pi / 2, N + 2 * N_ae)
        x_work_b[N - N_ae:2 * N + N_ae] = R * cos(t_work_prime) - ae
        y_work_b[N - N_ae:2 * N + N_ae] = R * sin(t_work_prime) - ae
        x_work_b[:N - N_ae] = [x_work_b[N - N_ae]] * (N - N_ae)
        y_work_b[:N - N_ae] = np.linspace(y_lim, y_work_b[N - N_ae], N - N_ae)
        x_work_b[2 * N + N_ae:] = np.linspace(x_work_b[2 * N + N_ae - 1], x_lim, N - N_ae)
        y_work_b[2 * N + N_ae:] = [y_work_b[2 * N + N_ae - 1]] * (N - N_ae)

        x_path_b[N:2 * N] = abs(R - r) * cos(t_work) - ae
        y_path_b[N:2 * N] = abs(R - r) * sin(t_work) - ae


    x_path_a[N:2 * N] = (R - r) * cos(t_work)
    y_path_a[N:2 * N] = (R - r) * sin(t_work)
    x_path_a[:N] = [x_path_a[N]] * N
    x_path_a[2 * N:] = np.linspace(x_path_a[2 * N - 1], x_lim, N)
    y_path_a[:N] = np.linspace(y_lim, y_path_a[N], N)
    y_path_a[2 * N:] = [y_path_a[2 * N - 1]] * N

    x_path_b[:N] = [x_path_b[N]] * N
    x_path_b[2 * N:] = np.linspace(x_path_b[2 * N - 1], x_lim, N)
    y_path_b[:N] = np.linspace(y_lim, y_path_b[N], N)
    y_path_b[2 * N:] = [y_path_b[2 * N - 1]] * N

    work_a = np.array([x_work_a, y_work_a])
    work_b = np.array([x_work_b, y_work_b])
    path_a = np.array([x_path_a, y_path_a])
    path_b = np.array([x_path_b, y_path_b])

    calc_lim = abs(x_lim)

    return work_a, work_b, path_a, path_b, calc_lim
# %%
def arc_intersection_b_R(R, r, ae, N, path_a, calc_lim, type):
    if type == 'concentric' and R - ae >= r:
        print('concentric')
        y_calc_start = r * sin(arccos(1 - ae / r))
        y_calc_end = ((R - r) ** 2 + (R - ae) ** 2 - r ** 2) / (2 * (R - ae))
        N_calc_start = int(N * y_calc_start / calc_lim)
        N_calc_start = N - N_calc_start
        diff_end = abs(path_a[1] - y_calc_end)
        N_calc_end = np.where(diff_end == min(diff_end))[0][0]
        path_a_calc = path_a[:, N_calc_start:N_calc_end].copy()

        K = ((R - ae) ** 2 - r ** 2 + path_a_calc[0] ** 2 + path_a_calc[1] ** 2) / 2
        A = np.where(path_a_calc[1] == 0, K, 1 + path_a_calc[0] ** 2 / path_a_calc[1] ** 2)
        B = np.where(path_a_calc[1] == 0, K, -2 * path_a_calc[0] * K / path_a_calc[1] ** 2)
        C = np.where(path_a_calc[1] == 0, K, (K / path_a_calc[1]) ** 2 - (R - ae) ** 2)
        p_b_R_x = np.where(path_a_calc[1] < 0, (-B + sqrt(B ** 2 - 4 * A * C)) / (2 * A), K)
        p_b_R_x = np.where(path_a_calc[1] > 0, (-B - sqrt(B ** 2 - 4 * A * C)) / (2 * A), p_b_R_x)
        p_b_R_x = np.where(path_a_calc[1] == 0, K / path_a_calc[0], p_b_R_x)
        p_b_R_y = np.where(path_a_calc[1] == 0, K, -(path_a_calc[0] / path_a_calc[1]) * p_b_R_x + K / path_a_calc[1])
        p_b_R_y = np.where(path_a_calc[1] == 0, sqrt((R - ae) ** 2 - p_b_R_x ** 2), p_b_R_y)

    elif type == 'inner' and R - ae >= r:
        print('inner')
        y_calc_start = r * sin(arccos(1 - ae / r)) + ae
        K = (r ** 2 - (R - r) ** 2 - ae ** 2 - (R - ae) ** 2) / 2
        A = (1 + ((R - ae) / ae) ** 2)
        B = 2 * (R - ae) * K / ae ** 2
        C = (K / ae) ** 2 - (R - r) ** 2
        y_calc_end = (-B + sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        N_calc_start = int(N * y_calc_start / calc_lim)
        N_calc_start = N - N_calc_start
        diff_end = abs(path_a[1] - y_calc_end)
        N_calc_end = np.where(diff_end == min(diff_end))[0][0]
        path_a_calc = path_a[:, N_calc_start:N_calc_end].copy()
        
        C_x_calc = path_a_calc[0]
        C_y_calc = path_a_calc[1]
        m = -((C_x_calc + ae) / (C_y_calc + ae))
        K = (R ** 2 - r ** 2 - 2 * ae ** 2 + C_x_calc ** 2 + C_y_calc ** 2) / (2 * (C_y_calc + ae))
        A = (m ** 2 + 1)
        B = 2 * (m * (K - C_y_calc) - C_x_calc)
        C = (K - C_y_calc) ** 2 + C_x_calc ** 2 - r ** 2

        p_b_R_x = np.where(C_y_calc <= -ae, (-B + sqrt(B ** 2 - 4 * A * C)) / (2 * A), C_y_calc)
        p_b_R_x = np.where(C_y_calc > -ae, (-B - sqrt(B ** 2 - 4 * A * C)) / (2 * A), p_b_R_x)
        p_b_R_y = m * p_b_R_x + K
   
    elif R - ae < r:
        y_calc_start = r * sin(arccos(1 - ae / r)) + ae
        if R == r:
            A = 1
            B = -2 * (R - ae)
            C = (R - ae) ** 2 - r ** 2 + ae ** 2
            y_calc_end = (-B - sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        else:
            y_calc_end = R - ae - sqrt(r ** 2 - (R - r - ae) ** 2)

        N_calc_start = int(N * y_calc_start / calc_lim)
        N_calc_start = N - N_calc_start
        diff_end = abs(path_a[1] - y_calc_end)
        N_calc_end = np.where(diff_end == min(diff_end))[0][0]
        path_a_calc = path_a[:, N_calc_start:N_calc_end].copy()

        C_x_calc = path_a_calc[0]
        C_y_calc = path_a_calc[1]
        m = -((C_x_calc + ae) / (C_y_calc + ae))
        K = (R ** 2 - r ** 2 - 2 * ae ** 2 + C_x_calc ** 2 + C_y_calc ** 2) / (2 * (C_y_calc + ae))
        A = (m ** 2 + 1)
        B = 2 * (m * (K - C_y_calc) - C_x_calc)
        C = (K - C_y_calc) ** 2 + C_x_calc ** 2 - r ** 2

        p_b_R_x = np.where(C_y_calc <= -ae, (-B + sqrt(B ** 2 - 4 * A * C)) / (2 * A), C_y_calc)
        p_b_R_x = np.where(C_y_calc > -ae, (-B - sqrt(B ** 2 - 4 * A * C)) / (2 * A), p_b_R_x)
        p_b_R_y = m * p_b_R_x + K

    return N_calc_start, N_calc_end, p_b_R_x, p_b_R_y

def arc_intersection(R, r, ae, N, calc_lim, type, work_a, work_b, path_a):
    alpha = arccos(1 - ae / r)
    p_b_y_0 = r * sin(alpha)
    p_b_y_0 += work_b[1][0]
    diff_b = abs(work_b[1] - p_b_y_0)
    i_b_0 = np.where(diff_b == min(diff_b))[0][0]
    P_a = work_a[:, :-i_b_0].copy()
    P_b = work_b[:, i_b_0:].copy()
    N_calc_start, N_calc_end, p_b_R_x, p_b_R_y = arc_intersection_b_R(R, r, ae, N, path_a, calc_lim, type)
    P_b[:, N_calc_start:N_calc_end] = np.array([p_b_R_x, p_b_R_y])
    p_b_out_x = path_a[0, N_calc_end:2 * N] - sqrt(r ** 2 - (R - ae - path_a[1, N_calc_end:2 * N]) ** 2)
    p_b_out_y = [R - ae] * len(p_b_out_x)
    P_b[:, N_calc_end:2 * N] = np.array([p_b_out_x, p_b_out_y])
    path_a = path_a[:, :-i_b_0]

    return P_a, P_b, path_a, i_b_0

def arc_length(r, P_a, P_b, path_a):
    vec_a = np.array([P_a[0] - path_a[0], P_a[1] - path_a[1]])
    vec_b = np.array([P_b[0] - path_a[0], P_b[1] - path_a[1]])
    inner = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]
    L_a = sqrt(vec_a[0] ** 2 + vec_a[1] ** 2)
    L_b = sqrt(vec_b[0] ** 2 + vec_b[1] ** 2)
    beta = arccos(inner / (L_a * L_b))
    arc_length = r * beta
    
    return arc_length

def plot_contact_arc_max(R, r, ae, type, work_a, work_b, t_tool, path_a, path_b, P_a, P_b, contact_arc, n = None, path_b_plot = False):
    if not n:
        N_max = np.where(contact_arc == max(contact_arc))[0][0]
        n = N_max
    x_tool = r * cos(t_tool) + path_a[0][n]
    y_tool = r * sin(t_tool) + path_a[1][n]
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_aspect('equal')
    ax.plot(work_a[0], work_a[1], lw = 3, c = 'k')
    ax.plot(work_b[0], work_b[1], lw = 3, c = 'gray', linestyle = '--')
    ax.plot(path_a[0], path_a[1], c = 'lightblue', lw = 3, linestyle = (0, (20, 2, 2, 2, 2, 2)), alpha = 0.5)
    if path_b_plot:
        ax.plot(path_b[0], path_b[1], c = 'orange', lw = 3, linestyle = (0, (20, 2, 2, 2, 2, 2)), alpha = 0.5)
    ax.plot(P_a[0, n], P_a[1, n], marker = 'o', markersize = 10, c = 'r', markeredgecolor = 'k', zorder = 10)
    ax.plot(P_b[0, n], P_b[1, n], marker = 'o', markersize = 10, c = 'r', markeredgecolor = 'k', zorder = 10)
    ax.plot(x_tool, y_tool, c = 'b')
    ax.vlines(
        x = 0,
        ymin = - R * 0.2,
        ymax = R * 1.1,
        color = 'k',
        lw = 0.5,
        linestyle = (0, (100, 10, 10, 10))
    )
    ax.hlines(
        y = 0,
        xmin = - R * 0.2,
        xmax = R * 1.1,
        color = 'k',
        lw = 0.5,
        linestyle = (0, (100, 10, 10, 10))
    )
    if type == 'inner' or R - ae < r:
        ax.vlines(
            x = -ae,
            ymin = - R * 0.2 - ae,
            ymax = R * 1.1 - ae,
            color = 'k',
            lw = 0.5,
            linestyle = (0, (100, 10, 10, 10))
        )
        ax.hlines(
            y = -ae,
            xmin = - R * 0.2 - ae,
            xmax = R * 1.1 - ae,
            color = 'k',
            lw = 0.5,
            linestyle = (0, (100, 10, 10, 10))
        )
    ax.vlines(x = path_a[0, n], ymin = path_a[1, n] - (r * 1.1), ymax = path_a[1, n] + (r * 1.1), color = 'k', lw = 0.5)
    ax.hlines(y = path_a[1, n], xmin = path_a[0, n] - (r * 1.1), xmax = path_a[0, n] + (r * 1.1), color = 'k', lw = 0.5)
    plot_format(
        ax,
        [-2 * R, R * 1.1],
        [-2 * R, R *  1.1],
        fontsize = 0,
        flame_width = 0,
        scale_length = 0,
        grid_width = 0
    )

def plot_contact_arc(R, D, z, Vc, fz, N, work_a, path_a, i_b_0, contact_arc):
    r = D / 2
    rpm = int((1e3 * Vc) / (np.pi * D))
    Vf = fz * z * rpm
    Vf /= 60
    L_path_in = abs(min(path_a[1]))
    L_path_R = (R - r) * pi / 2
    L_path_out = abs(min(path_a[0]))

    T_in = L_path_in / Vf
    T_R = L_path_R / Vf + T_in
    T_out = L_path_out / Vf + T_R
    T = np.zeros(len(path_a[0]), dtype = np.float32)
    T[:N] = np.linspace(0, T_in, N)
    T[N:2 * N] = np.linspace(T_in, T_R, N)
    T[2 * N:] = np.linspace(T_R, T_out, N - i_b_0)
    N_R_start = np.where(work_a[1] == 0)[0][0]
    T = T - T[N_R_start]
    fig, ax = plt.subplots()
    ax.plot(T, contact_arc, c = 'k')
    plot_format(
    ax,
    [-max(T / 3), max(T / 3)],
    [0, 1.1 * max(contact_arc)],
    fontsize = 15,
    flame_width = 2,
    scale_length = 8,
    pad = [5, 3]
    )

    return T, fig, ax

def contact_arc_anime(R, D, r, ae, type, N, work_a, work_b, path_a, t_tool, i_b_0, P_a, P_b, contact_arc, T, dpi):
    N_anime = int(N / 100)
    N_anime = np.arange(0, (3 * N) - 1, N_anime, dtype = 'int')
    fig, ax = plt.subplots(figsize = (10, 10), dpi = dpi)
    ax.set_aspect('equal')
    ax.plot(work_a[0, N_anime], work_a[1, N_anime], lw = 3, c = 'k')        
    ax.plot(work_b[0, N_anime], work_b[1, N_anime], lw = 3, c = 'gray')
    N_anime = int(N / 100)
    N_anime = np.arange(0, (3 * N - i_b_0) - 1, N_anime, dtype = 'int')
    ax.plot(path_a[0, N_anime], path_a[1, N_anime], c = 'blue', lw = 3, linestyle = (0, (20, 2, 2, 2, 2, 2)), alpha = 0.3)
    ax.text(0.5, 0.05, f'R:{R} mm, D:{D} mm, Ae:{ae} mm', ha='center', transform=ax.transAxes, fontsize = 30)
    ax.vlines(
        x = 0,
        ymin = - R * 0.2,
        ymax = R * 1.1,
        color = 'k',
        lw = 0.5,
        linestyle = (0, (100, 10, 10, 10))
    )
    ax.hlines(
        y = 0,
        xmin = - R * 0.2,
        xmax = R * 1.1,
        color = 'k',
        lw = 0.5,
        linestyle = (0, (100, 10, 10, 10))
    )
    if type == 'inner' or R - ae < r:
        ax.vlines(
            x = -ae,
            ymin = - R * 0.2 - ae,
            ymax = R * 1.1 - ae,
            color = 'k',
            lw = 0.5,
            linestyle = (0, (100, 10, 10, 10))
        )
        ax.hlines(
            y = -ae,
            xmin = - R * 0.2 - ae,
            xmax = R * 1.1 - ae,
            color = 'k',
            lw = 0.5,
            linestyle = (0, (100, 10, 10, 10))
        )
    plot_format(
        ax,
        [-3 * R, R * 1.1],
        [-3 * R, R *  1.1],
        fontsize = 10,
        flame_width = 0,
        scale_length = 0,
        pad = [5, 5],
        grid_color = 'lightblue',
        grid_width = 0.5,
        grid_which = 'both',
        minor_grid_axis = 'both'
    )
    T_right = abs(T - (max(T) / 3))
    n_T_right = np.where(T_right == min(T_right))[0][0]
    T_left = abs(T + (max(T) / 3))
    n_T_left = np.where(T_left == min(T_left))[0][0]
    N_anime = int(N / 50)
    N_anime = np.linspace(n_T_left, n_T_right, N_anime, dtype = int)
    T_anime = np.round(T[N_anime], decimals = 3)
    contact_arc_anime = np.round(contact_arc[N_anime], decimals = 3)
    path_a_anime = path_a[:, N_anime]
    P_a_anime = P_a[:, N_anime]
    P_b_anime = P_b[:, N_anime]
    vec_stn = np.zeros([2, len(N_anime)])
    vec_stn[0] = [1] * len(N_anime)
    vec_CPa = P_a_anime - path_a_anime
    vec_CPb = P_b_anime - path_a_anime
    inner_a = vec_CPa[0] * vec_stn[0] + vec_CPa[1] * vec_stn[1]
    inner_b = vec_CPb[0] * vec_stn[0] + vec_CPb[1] * vec_stn[1]
    L_a = sqrt(vec_CPa[0] ** 2 + vec_CPa[1] ** 2)
    L_b = sqrt(vec_CPb[0] ** 2 + vec_CPb[1] ** 2)
    L_stn = sqrt(vec_stn[0] ** 2 + vec_stn[1] ** 2)
    beta_a = arccos(inner_a / (L_a * L_stn))
    beta_b = arccos(inner_b / (L_b * L_stn))
    frames = []
    for n in range(len(N_anime)):
        t_arc = np.linspace(beta_a[n], beta_b[n], 100)
        C_x = path_a_anime[0, n]
        C_y = path_a_anime[1, n]
        x_tool = r * cos(t_tool) + C_x
        y_tool = r * sin(t_tool) + C_y
        x_arc = r * cos(t_arc) + C_x
        y_arc = r * sin(t_arc) + C_y
        frame = ax.plot(P_a_anime[0, n], P_a_anime[1, n], marker = 'o', markersize = 10, c = 'r', markeredgecolor = 'k', zorder = 10)
        frame += ax.plot(P_b_anime[0, n], P_b_anime[1, n], marker = 'o', markersize = 10, c = 'r', markeredgecolor = 'k', zorder = 10)
        frame += ax.plot(x_tool, y_tool, c = 'b')
        frame += ax.plot(x_arc, y_arc, c = 'red', lw = 5)
        frame += [ax.vlines(x = path_a_anime[0, n], ymin = path_a_anime[1, n] - (r * 1.1), ymax = path_a_anime[1, n] + (r * 1.1), color = 'k', lw = 0.5)]
        frame += [ax.hlines(y = path_a_anime[1, n], xmin = path_a_anime[0, n] - (r * 1.1), xmax = path_a_anime[0, n] + (r * 1.1), color = 'k', lw = 0.5)]
        frame += [ax.text(0.5, 1.1, f'Time:{T_anime[n]} s', ha='center', transform=ax.transAxes, fontsize = 30)]
        frame += [ax.text(0.5, 1.05, f'Contact Arc:{contact_arc_anime[n]} mm', ha='center', transform=ax.transAxes, fontsize = 30)]
        frames.append(frame)
    anime = ArtistAnimation(fig, frames, interval = 200)
    os.makedirs('contact_arc', exist_ok = True)
    anime.save(f'contact_arc/R{R}_D{D}_ae{ae}.gif', writer = 'pillow')

def contact_arc(R, D, z, ae, Vc, fz, type = 'concentric', N = 1000, n = None, path_b_plot = False, anime = False, dpi = 60, result = 'only'):
    start = time.time()
    r = D / 2
    t_tool = np.linspace(0, 2 * pi, 100)
    work_a, work_b, path_a, path_b, calc_lim = arc_coordinate(R, D, ae, N, type = type)
    P_a, P_b, path_a, i_b_0 = arc_intersection(R, r, ae, N, calc_lim, type, work_a, work_b, path_a)
    contact_arc = arc_length(r, P_a, P_b, path_a)
    end = time.time()
    print(f'接触弧長さ演算時間:{np.round(end - start, decimals = 3)} s')
    T, fig, ax = plot_contact_arc(R, D, z, Vc, fz, N, work_a, path_a, i_b_0, contact_arc)
    plot_contact_arc_max(R, r, ae, type, work_a, work_b, t_tool, path_a, path_b, P_a, P_b, contact_arc, n, path_b_plot)
    if anime:
        contact_arc_anime(R, D, r, ae, type, N, work_a, work_b, path_a, t_tool, i_b_0, P_a, P_b, contact_arc, T, dpi)
    ans = pd.DataFrame(
        {
            'time':T,
            'contact_arc':contact_arc
        }
    )
    if result == 'only':
        return ans
    if result == 'all':
        return ans, work_a, work_b, path_a, path_b, P_a, P_b, i_b_0
def simulation_anime(
    R, D, z, ap, ae, Vc, fz,
    N, n_T_left, n_T_right,
    work_a, work_b, path_a, x_edge, y_edge,
    i_b_0, P_a, P_b, P_x, P_y, Q_x, Q_y,
    L, T, L_contact_edge_sum, N_contact_edge,
    dpi
):
    r = D / 2
    if R - r == 0:
        N_R_prime = [N + 1]
    else:
        l_path_in = 3 * R  
        l_path_R = (R - r) * pi / 2
        in_by_R = int(np.round(l_path_in / l_path_R))
        N_R_prime = np.arange(1, N - 1, in_by_R, dtype = int)
        N_R_prime += N
        N_R_prime = list(N_R_prime)
    N_anime = list(range(n_T_left, N))
    N_anime += N_R_prime
    N_anime += list(range(2 * N, n_T_right))
    vec_stn = np.zeros([2, len(T)], dtype = np.float32)
    vec_stn[0] = [1] * len(T)
    vec_CPa = P_a - path_a
    vec_CPb = P_b - path_a
    inner_a = vec_CPa[0] * vec_stn[0] + vec_CPa[1] * vec_stn[1]
    inner_b = vec_CPb[0] * vec_stn[0] + vec_CPb[1] * vec_stn[1]
    L_a = sqrt(vec_CPa[0] ** 2 + vec_CPa[1] ** 2)
    L_b = sqrt(vec_CPb[0] ** 2 + vec_CPb[1] ** 2)
    L_stn = sqrt(vec_stn[0] ** 2 + vec_stn[1] ** 2)
    beta_a = arccos(inner_a / (L_a * L_stn))
    beta_b = arccos(inner_b / (L_b * L_stn))
    beta_a = np.float32(beta_a)
    beta_b = np.float32(beta_b)
    T_round = np.round(T, decimals = 3)
    
    
    
    fig = plt.figure(dpi = dpi)
    ax1 = fig.add_subplot(3, 2, (1, 3))
    ax2 = fig.add_subplot(3, 2, 5)
    ax3 = fig.add_subplot(3, 2, 2)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 6)
    def update(f):
        n = N_anime[f]
        ax1.cla()
        calc.work_plot(ax1, n, R, D, z, ap, ae, Vc, fz, work_a, work_b, path_a, N, T_round, i_b_0)
        calc.tool_plot(ax1, r, path_a, P_a, P_b, beta_a, beta_b, n)
        ax2.cla()
        calc.plot_format(
                ax2,
                [-1, math.ceil(1.1 * max(L))],
                [0, math.ceil(1.1 * ap)]
        )
        ax2.set_aspect('equal')
        L_n = L[n]
        L_n = np.linspace(0, L_n, 10)
        ax2.fill_between(L_n, 0, ap, color = 'deepskyblue', alpha = 0.5)
        for z_i in range(z):
            ax2.plot(x_edge[n, z_i], y_edge, c = 'k')
            ax2.plot(P_x[n, z_i], P_y[n, z_i], marker = 'o', markersize = 5, markeredgecolor = 'k', c = 'r', zorder = 10)
            ax2.plot(Q_x[n, z_i], Q_y[n, z_i], marker = 'o', markersize = 5, markeredgecolor = 'k', c = 'r', zorder = 10)
        ax3.cla()
        ax3.set_ylabel('接触弧長さ mm', fontname = 'MS Mincho')
        ax3.plot(T[:n], L[:n], c = 'k')
        ax3.plot(T[n], L[n], c = 'r', marker = 'o')
        calc.plot_format(ax3, [T[n] - 0.05, T[n] + 0.005], [0, max(L) + int(1)])
        ax4.cla()
        ax4.set_ylabel('切れ刃長さ mm', fontname = 'MS Mincho')
        ax4.plot(T[:n], L_contact_edge_sum[:n], c = 'k')
        ax4.plot(T[n], L_contact_edge_sum[n], c = 'r', marker = 'o')
        calc.plot_format(ax4, [T[n] - 0.05, T[n] + 0.005], [0, max(L_contact_edge_sum) * 1.05])
        ax5.cla()
        ax5.set_xlabel('時間 s', fontname = 'MS Mincho')
        ax5.set_ylabel('同時切削刃数', fontname = 'MS Mincho')
        ax5.plot(T[:n], N_contact_edge[:n], c = 'k')
        ax5.plot(T[n], N_contact_edge[n], c = 'r', marker = 'o')
        calc.plot_format(ax5, [T[n] - 0.05, T[n] + 0.005], [0, int(max(N_contact_edge)) + int(1)])
    k = len(glob.glob('simulate/*.gif'))
    anime = FuncAnimation(fig, update, frames = tqdm.tqdm(range(len(N_anime)), desc = f'simu{k}'), interval = 20)
    os.makedirs('simulate', exist_ok = True)
    filename = f'R{R}D{D}z{z}Ap{ap}Ae{ae}Vc{Vc}fz{fz}'
    filename = filename.replace('.', '')
    anime.save(f'simulate/{filename}.gif', writer = "pillow")
    movie_file = mp.VideoFileClip(f'simulate/{filename}.gif')
    movie_file.write_videofile(f'simulate/{filename}.mp4')
    movie_file.close()
# %%
def simulate(
    R, D, z, ap, ae,
    N = 1000, type = 'inner',
    theta = 38, L_edge = 6, Vc = 90, fz = 0.03, initial_phase = 0,
    n_plot = None, anime = False, dpi = 100
):
    start = time.time()
    print('演算開始')
    df_contact_arc, work_a, work_b, path_a, path_b, P_a, P_b, i_b_0 = contact_arc(
        R, D, z, ae, Vc, fz,
        type, N,
        n_plot, path_b_plot = False, anime = False, result = 'all'
        )
    print(f'接触弧長さ演算完了{np.round(time.time() - start, decimals = 3)}s')

    r = D / 2
    p = pi * D / z
    T = df_contact_arc['time'].values
    L = df_contact_arc['contact_arc'].values

    T_right = abs(T - (max(T) / 3))
    n_T_right = np.where(T_right == min(T_right))[0][0]
    T_left = abs(T + (max(T) / 3))
    n_T_left = np.where(T_left == min(T_left))[0][0]

    N_prime = len(T)
    omega_c = (1e3 * Vc / 60) / r
    alpha = omega_c * T
    alpha = np.diff(alpha)
    trans = list(range(1, z))
    trans.append(0)

    S_1 = -abs(ap * tan(radians(theta), dtype = np.float32))
    S_2 = 0
    S_4 = L.reshape([N_prime, 1])
    S_3 = S_4 - ap * tan(radians(theta), dtype = np.float32)
    n_negativ_edge = int(S_1 / p - 1)
    S_0 = p * n_negativ_edge

    x_max_0 = L_edge * tan(radians(theta), dtype = np.float32)
    x_edge_0 = np.zeros([z, 10], dtype = np.float32)
    for i in range(n_negativ_edge, z + n_negativ_edge):
        x_edge_0[i - n_negativ_edge] = np.linspace(p * i, x_max_0 + (p * i), 10)
    x_edge_0 -= r * np.radians(initial_phase)
    y_edge = np.linspace(0, L_edge, 10)
    x_edge = np.zeros([N_prime, z, 10])
    x_edge[0] = x_edge_0
    for n in range(1, N_prime):
        x_edge[n] = x_edge[n - 1] - r * sin(alpha[n - 1])
        x_judge = x_edge[n, 0, 0]
        if x_judge < S_0:
            x_edge[n] = x_edge[n, trans]
            x_edge[n, -1] += pi * D

    x_edge = np.round(x_edge, decimals = 6)
    S_0 = np.round(S_0, decimals = 6)
    S_1 = np.round(S_1, decimals = 6)
    S_3 = np.round(S_3, decimals = 6)
    S_4 = np.round(S_4, decimals = 6)

    P_x = x_edge[:, :, 0]
    P_y = np.zeros([N_prime, z])
    Q_x = P_x + ap * tan(radians(theta))
    Q_y = np.zeros([N_prime, z])
    P_y = np.where((S_1 <= P_x) & (P_x < S_2), abs(P_x) / tan(radians(theta)), P_y)
    P_y = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, P_y)
    Q_x = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, Q_x)
    Q_x = np.where((S_3 <= P_x) & (P_x < S_4), S_4, Q_x)
    Q_y = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, Q_y)
    Q_y = np.where((S_1 <= P_x) & (P_x < S_3), ap, Q_y)
    Q_y = np.where((S_3 <= P_x) & (P_x < S_4), (S_4 - P_x) / tan(radians(theta)), Q_y)
    P_x = np.where((S_1 <= P_x) & (P_x < S_2), 0, P_x)
    P_x = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, P_x)

    P_x = np.round(P_x, decimals = 6)
    P_y = np.round(P_y, decimals = 6)
    Q_x = np.round(Q_x, decimals = 6)
    Q_y = np.round(Q_y, decimals = 6)

    L_contact_edge = sqrt((Q_x - P_x) ** 2 + (Q_y - P_y) ** 2, dtype = np.float32)
    L_contact_edge_sum = np.nansum(L_contact_edge, axis = 1)
    N_contact_edge = np.nan_to_num(L_contact_edge)
    N_contact_edge = np.where((abs(P_y - ap) < 6e-3) & (abs(Q_x) < 4.5e-3), 0, N_contact_edge)
    N_contact_edge = np.count_nonzero(N_contact_edge, axis=1)
    N_contact_edge = np.int32(N_contact_edge)
    ans = pd.DataFrame(
        {
            'time':T,
            'L_contact_arc':L,
            'N_SCE':N_contact_edge,
            'L_SCE':L_contact_edge_sum
        }
    )
    if R - r == 0:
        N_R_true = [N + 1]
    else:
        l_path_in = 3 * R  
        l_path_R = (R - r) * pi / 2
        in_by_R = int(np.round(l_path_in / l_path_R))
        N_R_true = np.arange(1, N - 1, in_by_R, dtype = int)
        N_R_true += N
        N_R_true = list(N_R_true)
    N_True = list(range(N))
    N_True += N_R_true
    N_True += list(range(2 * N, len(ans)))
    ans = ans.iloc[N_True]
    ans_time_0 = ans.loc[ans['time'] == 0]
    if len(ans_time_0) > 1:
        ans = ans.drop(ans_time_0.index.values[1:])
    ans = ans.reset_index(drop = True)
    print(f'シミュレーション演算終了{np.round(time.time() - start, decimals = 3)}s')

    if not n_plot:
        n_plot = np.where(L_contact_edge_sum == max(L_contact_edge_sum))[0][0]
    L_n = L[n_plot]
    L_n = np.linspace(0, L_n, 10)
    fig, ax = plt.subplots()
    ax.fill_between(L_n, 0, ap, color = 'deepskyblue', alpha = 0.5)
    ax.set_aspect('equal')
    calc.plot_format(
            ax,
            [-1, math.ceil(1.1 * max(L))],
            [0, math.ceil(1.1 * ap)]
    )
    for z_i in range(z):
        ax.plot(P_x[n_plot, z_i], P_y[n_plot, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
        ax.plot(Q_x[n_plot, z_i], Q_y[n_plot, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
        ax.plot(x_edge[n_plot, z_i], y_edge, c = 'k')

    fig, ax = plt.subplots()
    ax.plot(T, L_contact_edge_sum, c = 'k')
    calc.plot_format(
            ax,
            [T[n_T_left], T[n_T_right]],
            [0, max(L_contact_edge_sum) * 1.1],
            fontsize = 15,
            pad = [5, 3]
    )
    fig, ax = plt.subplots()
    ax.plot(T, N_contact_edge, c = 'k')
    calc.plot_format(
            ax,
            [T[n_T_left], T[n_T_right]],
            [0, max(N_contact_edge) + 1],
            fontsize = 15,
            pad = [5, 3]
    )

    if anime:
        print(f'アニメーション開始{np.round(time.time() - start, decimals = 3)}s')
        simulation_anime(
            R, D, z, ap, ae, Vc, fz,
            N, n_T_left, n_T_right,
            work_a, work_b, path_a, x_edge, y_edge,
            i_b_0, P_a, P_b, P_x, P_y, Q_x, Q_y,
            L, T, L_contact_edge_sum, N_contact_edge,
            dpi
        )
    print(f'Accomplished all{np.round(time.time() - start, decimals = 3)}s')    
    return ans