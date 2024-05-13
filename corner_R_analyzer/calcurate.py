# %%
import numpy as np
import time
import math
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from numpy import sin, cos, tan, sqrt, arcsin, arccos, arctan, radians, degrees, pi
import os
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
    
def contact_arc_max(R, D, ae):
    r = D / 2
    inner_circle = np.round(R - r, decimals = 3) <= ae
    concentric_circle = np.round(R - r, decimals = 3) > ae
    
    A = np.where(inner_circle, R - r, ae)
    B = np.where(inner_circle, R ** 2 - r ** 2 + A ** 2 - 2 * ae ** 2, ae)
    l = np.where(inner_circle, (A ** 2) + ((B ** 2) / (4 * ae ** 2)) - (r ** 2), ae)
    m = np.where(inner_circle, 1 + ((ae + A) / ae) ** 2, ae)
    n = np.where(inner_circle, 2 * A + B * (ae + A) / ae ** 2, ae)

    x = np.where(inner_circle, (-n + np.sqrt(np.round(n ** 2 - 4 * m * l, decimals = 5))) / (2 * m), ae)
    alpha = np.where(inner_circle & (ae < x), A ** 2 + R ** 2 - 2 * R * ae - r ** 2 + ae ** 2, ae)
    X = np.where(inner_circle & (ae < x), -A + np.sqrt(np.round(A ** 2 - alpha, decimals = 5)), ae)
    L = np.where(inner_circle & (x <= ae), r * np.arccos(1 - (R + x) / r), ae)
    L = np.where(inner_circle & (ae < x), R * np.arccos(1 - (R + X) / r), L)

    R_prime = np.where(concentric_circle, R - ae, ae)
    r_tp = np.where(concentric_circle, R - r, ae)
    X_ce = np.where(concentric_circle, r_tp, ae)
    Y_ce = np.where(concentric_circle, 0, ae)

    x_a = np.where(concentric_circle, np.round((X_ce ** 2 + R ** 2 - r ** 2) / (2 * X_ce), decimals = 3), ae)
    y_a = np.where(concentric_circle, np.sqrt(np.round(R ** 2 - x_a ** 2, decimals = 5)), ae)
    x_b = np.where(concentric_circle, (X_ce ** 2 + R_prime ** 2 - r ** 2) / (2 * X_ce), ae)
    y_b = np.where(concentric_circle, np.sqrt(np.round(R_prime ** 2 - x_b ** 2, decimals = 5)), ae)


    vec_a_x = np.where(concentric_circle, x_a - X_ce, ae)
    vec_a_y = np.where(concentric_circle, y_a - Y_ce, ae)
    vec_b_x = np.where(concentric_circle, x_b - X_ce, ae)
    vec_b_y = np.where(concentric_circle, y_b - Y_ce, ae)
    
    inner = np.where(concentric_circle, vec_a_x * vec_b_x +  vec_a_y * vec_b_y, ae)
    l_vec_a = np.where(concentric_circle, np.sqrt(vec_a_x ** 2 + vec_a_y ** 2), ae)
    l_vec_b = np.where(concentric_circle, np.sqrt(vec_b_x ** 2 + vec_b_y ** 2), ae)
    theta = np.where(concentric_circle, np.arccos(inner / (l_vec_a * l_vec_b)), ae)

    L = np.where(concentric_circle, r * theta, L)
    L = np.round(L, decimals = 5)
    return L

def contact_arc_max_inner(R, D, ae):
    r = D / 2                 
    A = R - r
    B = R ** 2 - r ** 2 + A ** 2 - 2 * ae ** 2
    m = 1 + ((ae + A) / ae) ** 2
    n = 2 * A + B * (ae + A) / ae ** 2
    l = (A ** 2) + ((B ** 2) / (4 * ae ** 2)) - (r ** 2)

    x = (- n + np.sqrt(n ** 2 - 4 * m * l)) / (2 * m)
    alpha = np.where(ae < x, A ** 2 + R ** 2 - 2 * R * ae - r ** 2 + ae ** 2, ae)
    X = np.where(ae < x, -A + np.sqrt(A ** 2 - alpha), ae)

    L = np.where(x <= ae, r * np.arccos(1- (R + x) / r), ae)
    L = np.where(ae < x, R * np.arccos(1 - ((R + X) / r)), L)      
    L = np.round(L, decimals = 5)
    return L
# %%
def contact_arc(R, D, z, ae, dx = 0.05, anime = False, save_dir = '', Vc = 90, fz = 0.03):
    r = D / 2
    R_tool = R - r

    x_work = np.round(np.arange(-3 * R, 3 * R + dx, dx), decimals = 3)
    x_tool = np.round(np.arange(-R, R + dx, dx), decimals = 3)

    root_2 = np.sqrt(2)

    y_a = []
    y_b = []
    y_tool = []

    for x in x_work:
        if x <= -R / root_2:
            y_a.append(x + R * root_2)

        elif -R / root_2 < x < R / root_2:
            y_a.append(np.sqrt(R ** 2 - x ** 2))

        elif R / root_2 <= x:
            y_a.append(-x + R * root_2)     

    for x in x_work:
        if x <= -R / root_2:
            y_b.append(x + (R - ae) * root_2)

        elif -R / root_2 < x < R / root_2 and R - r < ae:
            y_b.append(np.sqrt(R ** 2 - x ** 2) - ae * root_2)
        elif -R / root_2 < x < R / root_2 and R - r >= ae:
            y_b.append(np.sqrt((R - ae) ** 2 - x ** 2))
        elif R / root_2 <= x:
            y_b.append(-x + (R - ae) * root_2)

    for x in x_tool:
        if x <= -R_tool / root_2:
            y_tool.append(x + R_tool * root_2)

        elif -R_tool / root_2 < x < R_tool / root_2:
            y_tool.append(np.sqrt(R_tool ** 2 - x ** 2))

        elif R_tool / root_2 <= x:
            y_tool.append(-x + R_tool * root_2)

    if anime:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xlim(-R - 5, R + 5)
        ax.set_ylim(-R - 5, R + 5)
        ax.plot(x_work, y_a, color = 'black')
        ax.plot(x_tool, y_tool, color = 'gray', linestyle = 'dashdot')

    df_work = pd.DataFrame(data = {'y_a':y_a, 'y_b':y_b}, index = x_work)
    df_tool = pd.DataFrame(data = {'y_tool':y_tool}, index = x_tool)
    df = pd.concat([df_work, df_tool], axis = 1)

    contact_arc = []
    ae_true = []
    frames = []

    for i in range(len(x_tool)):
        x_endmill = np.arange(-r, r + dx, dx)
        y_endmill = np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        y_endmill_fig = -np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        x_endmill = np.round(x_endmill + x_tool[i], decimals = 3)
        df_endmill = pd.DataFrame(data = {'x_endmill':x_endmill, 'y_endmill':y_endmill}, index = x_endmill)
        df_calc = pd.concat([df, df_endmill], axis = 1)
        df_calc['L_a'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_a']) ** 2)
        df_calc['L_b'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_b']) ** 2)
        P_x = df_calc['L_a'].idxmin()

        df_cut = df_calc.loc[P_x:]
        Q_x = df_cut['L_b'].idxmin()

        if not P_x  == 'nan':
            P_y = df_calc.loc[P_x, 'y_a']
            Q_y = df_calc.loc[Q_x, 'y_b']
            
            vec_P = [P_x - x_tool[i], P_y - y_tool[i]]
            vec_Q = [Q_x - x_tool[i], Q_y - y_tool[i]]
            inner = np.inner(vec_P, vec_Q)
            theta = np.arccos(inner / (np.linalg.norm(vec_P) * np.linalg.norm(vec_Q)))
            contact_arc.append([x_tool[i], r * theta])
            ae_true.append([x_tool[i], r * (1 - np.cos(theta))])
            if anime:
                frame_1 = ax.plot(x_endmill, y_endmill, color = 'blue')
                frame_2 = ax.plot(x_endmill, y_endmill_fig, color = 'blue')
                frame_5 = ax.plot(df_calc.loc[Q_x:].index, df_calc.loc[Q_x:, 'y_b'], color = 'gray')
                frame_3 = ax.plot(P_x, P_y, marker = '*', color = 'red', markersize = 10)
                frame_4 = ax.plot(Q_x, Q_y, marker = '*', color = 'red', markersize = 10)

                frames.append(frame_1 + frame_2 + frame_3 + frame_4 + frame_5)

        else:
            if anime:
                frames.append(frame_1 + frame_2)

    if anime:
        anime = ArtistAnimation(fig, frames, interval = dx * 10)
        os.makedirs(save_dir, exist_ok = True)
        save_dir += '/'
        anime.save(f'{save_dir}animation.gif', writer='pillow')

    n = int((1e3 * Vc) / (np.pi * D))
    Vf = fz * z * n
    Vf /= 60
    Vf /= root_2
    alpha = np.arccos(1 - ae / R)
    x_R_start = R * np.sin(alpha)
    x_R_start = max(x_work) - (R - r) / root_2 - x_R_start / root_2
    t_R_start = x_R_start / Vf
    t_start = 0
    t_end = max(x_work) * 2 / Vf
    time = np.linspace(t_start, t_end, len(contact_arc))
    time -= t_R_start
    contact_arc = pd.DataFrame(contact_arc, columns = ['tool_position', 'L'])
    contact_arc['time'] = time
    fig, ax = plt.subplots()
    ax.plot(contact_arc['tool_position'], contact_arc['L'], color = 'black')
    ax.set_xlim(min(contact_arc['tool_position']), max(contact_arc['tool_position']))
    ax.set_ylim(0, max(contact_arc['L']) + 1)
    ax.plot(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'], max(contact_arc['L']), marker = '.', markersize = 15, color = 'black')
    ax.text(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'] + 0.1, max(contact_arc['L']), str(np.round(max(contact_arc['L']), decimals = 2)))

    # ae_true = pd.DataFrame(ae_true, columns = ['tool_position', 'ae_true'])
    # fig, ax = plt.subplots()
    # ax.plot(ae_true['tool_position'], ae_true['ae_true'], color = 'black')
    # ax.set_xlim(min(ae_true['tool_position']), max(ae_true['tool_position']))
    # ax.set_ylim(0, max(ae_true['ae_true']) + 0.1)
    # ax.plot(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'], max(ae_true['ae_true']), marker = '.', markersize = 15, color = 'black')
    # ax.text(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'] + 0.1, max(ae_true['ae_true']), str(np.round(max(ae_true['ae_true']), decimals = 2)))

    return contact_arc
# %%
def contact_arc_inner(R, D, z, ae, dx = 0.05, anime = False, save_dir = '', Vc = 90, fz = 0.03):
    r = D / 2
    R_tool = R - r

    x_work = np.round(np.arange(-3 * R, 3 * R + dx, dx), decimals = 3)
    x_tool = np.round(np.arange(-R, R + dx, dx), decimals = 3)

    root_2 = np.sqrt(2)

    y_a = []
    y_b = []
    y_tool = []

    for x in x_work:
        if x <= -R / root_2:
            y_a.append(x + R * root_2)

        elif -R / root_2 < x < R / root_2:
            y_a.append(np.sqrt(R ** 2 - x ** 2))

        elif R / root_2 <= x:
            y_a.append(-x + R * root_2)     

    for x in x_work:
        if x <= -R / root_2:
            y_b.append(x + (R - ae) * root_2)

        elif -R / root_2 < x < R / root_2:
            y_b.append(np.sqrt(R ** 2 - x ** 2) - ae * root_2)

        elif R / root_2 <= x:
            y_b.append(-x + (R - ae) * root_2)

    for x in x_tool:
        if x <= -R_tool / root_2:
            y_tool.append(x + R_tool * root_2)

        elif -R_tool / root_2 < x < R_tool / root_2:
            y_tool.append(np.sqrt(R_tool ** 2 - x ** 2))

        elif R_tool / root_2 <= x:
            y_tool.append(-x + R_tool * root_2)

    if anime:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xlim(-R - 5, R + 5)
        ax.set_ylim(-R - 5, R + 5)
        ax.plot(x_work, y_a, color = 'black')
        ax.plot(x_tool, y_tool, color = 'gray', linestyle = 'dashdot')

    df_work = pd.DataFrame(data = {'y_a':y_a, 'y_b':y_b}, index = x_work)
    df_tool = pd.DataFrame(data = {'y_tool':y_tool}, index = x_tool)
    df = pd.concat([df_work, df_tool], axis = 1)

    contact_arc = []
    ae_true = []
    frames = []

    for i in range(len(x_tool)):
        x_endmill = np.arange(-r, r + dx, dx)
        y_endmill = np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        y_endmill_fig = -np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        x_endmill = np.round(x_endmill + x_tool[i], decimals = 3)
        df_endmill = pd.DataFrame(data = {'x_endmill':x_endmill, 'y_endmill':y_endmill}, index = x_endmill)
        df_calc = pd.concat([df, df_endmill], axis = 1)
        df_calc['L_a'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_a']) ** 2)
        df_calc['L_b'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_b']) ** 2)
        P_x = df_calc['L_a'].idxmin()

        df_cut = df_calc.loc[P_x:]
        Q_x = df_cut['L_b'].idxmin()

        if not P_x  == 'nan':
            P_y = df_calc.loc[P_x, 'y_a']
            Q_y = df_calc.loc[Q_x, 'y_b']
            
            vec_P = [P_x - x_tool[i], P_y - y_tool[i]]
            vec_Q = [Q_x - x_tool[i], Q_y - y_tool[i]]
            inner = np.inner(vec_P, vec_Q)
            theta = np.arccos(inner / (np.linalg.norm(vec_P) * np.linalg.norm(vec_Q)))
            contact_arc.append([x_tool[i], r * theta])
            ae_true.append([x_tool[i], r * (1 - np.cos(theta))])
            if anime:
                frame_1 = ax.plot(x_endmill, y_endmill, color = 'blue')
                frame_2 = ax.plot(x_endmill, y_endmill_fig, color = 'blue')
                frame_5 = ax.plot(df_calc.loc[Q_x:].index, df_calc.loc[Q_x:, 'y_b'], color = 'gray')
                frame_3 = ax.plot(P_x, P_y, marker = '*', color = 'red', markersize = 10)
                frame_4 = ax.plot(Q_x, Q_y, marker = '*', color = 'red', markersize = 10)

                frames.append(frame_1 + frame_2 + frame_3 + frame_4 + frame_5)

        else:
            if anime:
                frames.append(frame_1 + frame_2)

    if anime:
        anime = ArtistAnimation(fig, frames, interval = dx * 10)
        os.makedirs(save_dir, exist_ok = True)
        save_dir += '/'
        anime.save(f'{save_dir}animation.gif', writer='pillow')
    n = int((1e3 * Vc) / (np.pi * D))
    Vf = fz * z * n
    Vf /= 60
    Vf /= root_2
    alpha = np.arccos(1 - ae / R)
    x_R_start = R * np.sin(alpha)
    x_R_start = max(x_work) - (R - r) / root_2 - x_R_start / root_2
    t_R_start = x_R_start / Vf
    t_start = 0
    t_end = max(x_work) * 2 / Vf
    time = np.linspace(t_start, t_end, len(contact_arc))
    time -= t_R_start
    contact_arc = pd.DataFrame(contact_arc, columns = ['tool_position', 'L'])
    contact_arc['time'] = time
    fig, ax = plt.subplots()
    ax.plot(contact_arc['tool_position'], contact_arc['L'], color = 'black')
    ax.set_xlim(min(contact_arc['tool_position']), max(contact_arc['tool_position']))
    ax.set_ylim(0, max(contact_arc['L']) + 1)
    ax.plot(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'], max(contact_arc['L']), marker = '.', markersize = 15, color = 'black')
    ax.text(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'] + 0.1, max(contact_arc['L']), str(np.round(max(contact_arc['L']), decimals = 2)))

    # ae_true = pd.DataFrame(ae_true, columns = ['tool_position', 'ae_true'])
    # fig, ax = plt.subplots()
    # ax.plot(ae_true['tool_position'], ae_true['ae_true'], color = 'black')
    # ax.set_xlim(min(ae_true['tool_position']), max(ae_true['tool_position']))
    # ax.set_ylim(0, max(ae_true['ae_true']) + 0.1)
    # ax.plot(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'], max(ae_true['ae_true']), marker = '.', markersize = 15, color = 'black')
    # ax.text(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'] + 0.1, max(ae_true['ae_true']), str(np.round(max(ae_true['ae_true']), decimals = 2)))

    return contact_arc
# %%
def contact_area(R, D, z, ap, ae, theta = 38, dx = 0.05, save_dir = ''):
    start = time.time()
    Px, Py, Qx, Qy, L_each, L_sum, N, t, Px_org = L_cut_edge(R, D, z, ap, ae, dx = dx, show_fig = False)
    L = contact_arc_max(R, D, ae)

    x1 = Px_org
    x2 = Px_org + 20 * tan(radians(theta))
    y_edge = [0, 20]
    t_org = t.copy()
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0)
    text_x = (Px + Qx) / 2
    text_y = (Py + Qy) / 2
    frames = []
    fig = plt.figure(figsize = (5, 10))
    plt.rcParams['font.family'] = ['Times New Roman']
    fig_max = math.ceil(max([ap, L])) + 1
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_title('R = ' + str(R) + 'mm, 刃数 = ' + str(z) + ', Ap = ' + str(ap) + 'mm, Ae = ' + str(ae) + 'mm', font = 'MS Gothic', fontsize = 15)
    ax1.set_xlim(-1, fig_max)
    ax1.set_ylim(-1, fig_max)
    ax1.hlines(y = [0, ap], xmin = 0, xmax = L, colors = 'blue')
    ax1.vlines(x = [0, L], ymin = 0, ymax = ap, colors = 'blue')
    ax1.axvspan(xmin = 0, xmax = L, ymin = 1 / (fig_max + 1), ymax = (ap + 1) / (fig_max + 1), color = 'cyan')
    ax1.grid()
    t_lim = np.round(4 * dt[1], decimals = 5)
    ax2.set_xlim(-t_lim, t_lim)
    ax2.set_xticks([-t_lim, 0, t_lim], [-t_lim, 0, t_lim])
    ax2.grid()
    count = int(1)
    frames = []
    for i in range(N):
        frame1 = ax1.plot(Px[:, i], Py[:, i], marker = 'o', color = 'red', linewidth = 0, markersize = 10) 
        frame1 += ax1.plot(Qx[:, i], Qy[:, i], marker = 'o', color = 'red', linewidth = 0, markersize = 10)
        frame1 += [ax1.text(x = 0, y = -0.5, ha = 'left', va = 'center', s = 't = ' + '{:.6f}'.format(t_org[i]) + 's', fontsize = 'x-large', font = 'MS Gothic')]
        frame2 = ax2.plot(t, L_sum, c = 'red')
        frame2 += ax2.plot(0, L_sum[i], c = 'red', marker = 'o', markersize = 10)
        frame2 += [ax2.text(x = t_lim / 10, y = L_sum[i],
                        ha = 'left', va = 'top',
                        s = '切れ刃長さ:' + str(np.round(L_sum[i], decimals = 3)) + 'mm', fontsize = 'x-large', font = 'MS Gothic')]
        t = t - dt[i]
        for j in range(len(Px)):
            if -ap * tan(radians(theta)) <= x1[j, i] and x1[j, i] <= L:
                color = 'red'
                number = '[' + str(count) + ']'
                frame1 += [ax1.text(x = text_x[j, i], y = text_y[j, i],
                                    ha = 'center', va = 'center',
                                    s = number, fontsize = 'xx-large', font = 'MS Gothic')]
                frame2 += [ax2.text(x = t_lim / 10, y = L_each[j, i],
                                    ha = 'center', va = 'center',
                                    s = number, fontsize = 'xx-large', font = 'MS Gothic')]
                count += 1
            else:
                color = 'black'
            x = np.array([x1[j, i], x2[j, i]])
            frame1 += ax1.plot(x, y_edge, c = color)
            frame2 += ax2.plot(t, L_each[j], c = 'blue', linestyle = '--')
            frame2 += ax2.plot(0, L_each[j, i], c = 'blue', marker = 'o', markersize = 10)
        frame2 += [ax2.text(x = t_lim / 10, y = L_sum[i] * 0.9,
                        ha = 'left', va = 'top',
                        s = '同時切削刃数:' + str(count - 1), fontsize = 'x-large', font = 'MS Gothic')]
        frames.append(frame1 + frame2)
        count = 1
    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(0, ymax2)
    ax2.vlines(x = 0, ymin = 0, ymax = ymax2, colors = 'black')
    anime = ArtistAnimation(fig, frames, interval = 100)
    anime.save(save_dir + str(R).replace('.', '') + '_z' + str(z).replace('.', '') + '_Ap' + str(ap).replace('.', '') + '_Ae' + str(ae).replace('.', '') + '.gif', writer='pillow')
    end = time.time()
    print('描画時間:' + str(np.round(end - start, decimals = 5)) + 's')
# %%
def cut_volume(R, D, ae, dx = 0.01, n = 1, anime = False, save = False, answer = False):
    r = D / 2
    R_tool = R - r
    x_work = -np.round(np.arange(-3 * R, 3 * R + dx, dx), decimals = 3)
    x_tool = -np.round(np.arange(-0.5 * R, 0.5 * R + dx, dx), decimals = 3)
    root_2 = np.sqrt(2)

    y_a = []
    y_b = []
    y_tool = []
    for x in x_work:
        if x <= -R / root_2:
            y_a.append(x + R * root_2)

        elif -R / root_2 < x < R / root_2:
            y_a.append(np.sqrt(R ** 2 - x ** 2))

        elif R / root_2 <= x:
            y_a.append(-x + R * root_2)

    for x in x_work:
        if x <= -R / root_2:
            y_b.append(x + (R - ae) * root_2)

        elif -R / root_2 < x < R / root_2:
            y_b.append(np.sqrt(R ** 2 - x ** 2) - ae * root_2)

        elif R / root_2 <= x:
            y_b.append(-x + (R - ae) * root_2)

    for x in x_tool:
        if x <= -R_tool / root_2:
            y_tool.append(x + R_tool * root_2)

        elif -R_tool / root_2 < x < R_tool / root_2:
            y_tool.append(np.sqrt(R_tool ** 2 - x ** 2))

        elif R_tool / root_2 <= x:
            y_tool.append(-x + R_tool * root_2)

    df = pd.DataFrame(data = {'x_work':x_work, 'y_a':y_a, 'y_b':y_b})
    
    if anime:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xlim(-R - 5, R + 5)
        ax.set_ylim(-R - 5, R + 5)
        ax.plot(x_work, y_a, color = 'black')
        ax.plot(x_tool, y_tool, color = 'gray', linestyle = 'dashdot')
        ax.plot(x_work, y_b, color = 'gray', linestyle = 'dashdot')

    frames = []
    cut_volume = []
    contact_arc = []
    for i in range(len(x_tool) - n):
        P_x = []
        P_y = []
        Q_x = []
        Q_y = []
        L = []
        for j in range(2):
            c_x = x_tool[i + n * j]
            c_y = y_tool[i + n * j]
            df_calc = df.copy()
            df_calc['d_a'] = np.abs(np.sqrt((df_calc['x_work'] - c_x) ** 2 + (df_calc['y_a'] - c_y) ** 2) - r)
            df_calc['d_b'] = np.abs(np.sqrt((df_calc['x_work'] - c_x) ** 2 + (df_calc['y_b'] - c_y) ** 2) - r)
            idmin_a = df_calc['d_a'].idxmin()
            df_calc = df_calc.loc[idmin_a:, :]
            idmin_b = df_calc['d_b'].idxmin()
            Pi_x = df_calc.loc[idmin_a, 'x_work']
            Pi_y = df_calc.loc[idmin_a, 'y_a']
            Qi_x = df_calc.loc[idmin_b, 'x_work']
            Qi_y = df_calc.loc[idmin_b, 'y_b']
            vec_P = [Pi_x - x_tool[i], Pi_y - y_tool[i]]
            vec_Q = [Qi_x - x_tool[i], Qi_y - y_tool[i]]
            inner = np.inner(vec_P, vec_Q)
            theta = np.arccos(inner / (np.linalg.norm(vec_P) * np.linalg.norm(vec_Q)))
            P_x.append(Pi_x)
            P_y.append(Pi_y)
            Q_x.append(Qi_x)
            Q_y.append(Qi_y)
            L.append(r * theta)
        contact_arc.append(L[0])
        cut_volume.append(sum(L))
        
        if anime:
            x_endmill = np.arange(-r, r + dx, dx)
            y_endmill_1 = np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
            y_endmill_fig_1 = -np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
            x_endmill_1 = np.round(x_endmill + x_tool[i], decimals = 3)
            y_endmill_2 = np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i + n]
            y_endmill_fig_2 = -np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i + n]
            x_endmill_2 = np.round(x_endmill + x_tool[i + n], decimals = 3)
            frame = ax.plot(x_endmill_1, y_endmill_1, color = 'blue', linestyle = '--')
            frame = frame + ax.plot(x_endmill_1, y_endmill_fig_1, color = 'blue', linestyle = '--')
            frame = frame + ax.plot(x_endmill_2, y_endmill_2, color = 'blue')
            frame = frame + ax.plot(x_endmill_2, y_endmill_fig_2, color = 'blue')
            frame = frame + ax.plot(P_x[0], P_y[0], marker = 'o', color = 'red')
            frame = frame + ax.plot(Q_x[0], Q_y[0], marker = 'o', color = 'red')
            frame = frame + ax.plot(P_x[1], P_y[1], marker = 'o', color = 'red')
            frame = frame + ax.plot(Q_x[1], Q_y[1], marker = 'o', color = 'red')
            frames.append(frame)
    
    if anime:
        anime = ArtistAnimation(fig, frames, interval = 1)
        if save:
            anime.save('animation_S.gif', writer='pillow')

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0 ,1, len(contact_arc)), contact_arc, color = 'black', linestyle = '--', label = '接触弧長さ')
    ax.plot(np.linspace(0 ,1, len(cut_volume)), cut_volume, color = 'black', label = '切削体積')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom = 0)
    ax.set_title('単位Apあたり切削体積', fontname = 'MS Mincho')
    ax.legend(prop = {'family':'MS Mincho'}, framealpha = 1, edgecolor = 'white')
    ax.grid(color = 'black')
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.minorticks_on()
    ax.tick_params(which = 'major', axis = 'y', direction = 'in', labelsize = 12, width = 2, length = 8)
    ax.tick_params(which = 'minor', axis = 'y', direction = 'in', labelsize = 12, width = 2, length = 5)
    ax.tick_params(which = 'major', axis = 'x', direction = 'in', labelsize = 12, width = 2, length = 8)
    ax.tick_params(which = 'minor', axis = 'x', direction = 'in', labelsize = 12, width = 2, length = 5)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.show()
    
    if answer:
        return np.array(cut_volume), np.array(contact_arc)
# %%
def L_cut_edge(R, D, z, ap, ae, theta = 38, Vc = 60, dx = 0.0001, show_fig = True):
    start = time.time()
    
    Vc = Vc * 1000 / 60
    L = contact_arc_max(R, D, ae)
    p = pi * D / z
    theta = radians(theta)
    x_min = -ap * tan(theta)
    x_max = L
    t_max = (x_max - x_min) / Vc
    distans = x_max - x_min
    n_edge = math.ceil(2 * distans / p) + 1
    N = int(distans / dx)
    t = np.round(np.linspace(0, t_max, N, dtype = 'float16'), decimals = 5)
    zero = np.zeros([n_edge, N], dtype = 'float32')
    Px = zero.copy()
    Py = zero.copy()
    Px[0] = np.linspace(x_max + distans, x_min + distans, N, dtype = 'float32')
    for i in range(1, n_edge):
        Px[i] = Px[i - 1] - p
    Px_org = Px.copy()
    Py = np.where((Px < x_min) | (L < Px), np.nan, Py)
    Qx = Px - x_min
    Qx = np.where((Px < x_min) | (L < Px), np.nan, Qx)
    Qy = Py + ap
    Qy = np.where(L < Qx, ap - (np.abs(Qx) - L) / tan(theta), Qy)
    Qx = np.where(L < Qx, L, Qx)
    Py = np.where((x_min <= Px) & (Px < 0), np.abs(Px) / tan(theta), Py)
    Px = np.where((x_min <= Px) & (Px < 0), 0, Px)
    L_each = np.round(sqrt((Px - Qx) ** 2 + (Py - Qy) ** 2), decimals = 3)
    L_sum = np.sum(np.nan_to_num(L_each), axis = 0)
    if show_fig:
        fig, ax = plt.subplots()
        for i in range(len(L_each)):
            ax.plot(t, L_each[i], c = 'blue', linestyle = '--')
        ax.plot(t, L_sum, color = 'black')

    end = time.time()
    print('演算時間:' + str(np.round(end - start, decimals = 5)) + 's')
    return Px, Py, Qx, Qy, L_each, L_sum, N, t, Px_org
# %%
def ae_from_L(R, D, L, ae_lim = 1, type = 'nomal'):
    L = np.matrix(L).T
    all_ae = np.arange(1 / 1e4, ae_lim + 1 / 1e4, 1 / 1e4, dtype = 'float32')
    if type == 'nomal':
        all_L = contact_arc_max(R, D, all_ae)
    else:
        all_L = contact_arc_max_inner(R, D, all_ae)
    all_ae = np.matrix([all_ae] * len(L))
    all_L = np.matrix([all_L] * len(L))
    L_diff = np.abs(all_L - L)
    ae = all_ae[np.where(L_diff == np.min((L_diff), axis = 1))]
    L_diff = L_diff[np.where(L_diff == np.min((L_diff), axis = 1))]
    ae = np.array(ae[np.where(L_diff <= 1 / 100)])[0]
    return ae
# %%
def cut_depth(R, D, theta, z_lim = 10, ae_lim = 1, check = False, type = 'nomal'):
    theta_rad = radians(theta)
    ans = []
    for z in range(2, z_lim + 1):

        p = pi * D / z
        ap = np.round(p / tan(theta_rad), decimals = 3)

        L_min = p
        L_max = p * int(z / 2) + p
        L = np.arange(L_min, L_max, p)
        ae = ae_from_L(R, D, L, ae_lim, type = type)
        if len(ae):
            ae = np.round(min(ae), decimals = 3)
            ans.append([z, ap, ae])
            if check:
                contact_area(R, D, z, ap, ae, theta = theta, dx = 0.05)
    ans = pd.DataFrame(ans, columns = ['z', 'Ap', 'Ae'])
    ans = ans.set_index('z')
    return ans
# %%
def work_plot(ax, n, R, D, z, ap, ae, Vc, fz, work_a, work_b, path_a, N, T, i_b_0):
    r = D / 2
    N_omis = int(N / 100)
    N_omis = np.arange(0, (3 * N) - 1, N_omis, dtype = 'int')
    ax.set_aspect('equal')
    ax.plot(work_a[0, N_omis], work_a[1, N_omis], lw = 3, c = 'k')        
    ax.plot(work_b[0, N_omis], work_b[1, N_omis], lw = 3, c = 'gray')
    N_omis_path = int(N / 100)
    N_omis_path = np.arange(0, (3 * N - i_b_0) - 1, N_omis_path, dtype = 'int')
    ax.plot(path_a[0, N_omis_path], path_a[1, N_omis_path], c = 'blue', lw = 3, linestyle = (0, (10, 1, 1, 1, 1, 1)), alpha = 0.3)
    ax.text(0.5, 0.12, f'R:{R} mm, Ap:{ap} mm, Ae:{ae} mm', ha='center', transform=ax.transAxes, fontsize = 8)
    ax.text(0.5, 0.06, f'D:{D} mm, z:{z}', ha='center', transform=ax.transAxes, fontsize = 8)
    ax.text(0.5, 0, f'Vc:{Vc} m/min, fz:{fz} mm/tooth', ha='center', transform=ax.transAxes, fontsize = 8)
    ax.text(0.5, 1.1, f'Time:{T[n]} s', ha='center', transform=ax.transAxes, fontsize = 10)
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
def tool_plot(ax, r, path_a, P_a, P_b, beta_a, beta_b, n):
    t_work = np.linspace(0, 2 * pi, 40)
    t_arc = np.linspace(beta_a[n], beta_b[n], 20)
    x_tool = r * cos(t_work) + path_a[0, n]
    y_tool = r * sin(t_work) + path_a[1, n]
    x_arc = r * cos(t_arc) + path_a[0, n]
    y_arc = r * sin(t_arc) + path_a[1, n]
    ax.plot(P_a[0, n], P_a[1, n], marker = 'o', markersize = 5, c = 'r', markeredgecolor = 'k', zorder = 10)
    ax.plot(P_b[0, n], P_b[1, n], marker = 'o', markersize = 5, c = 'r', markeredgecolor = 'k', zorder = 10)
    ax.plot(x_tool, y_tool, c = 'b')
    ax.plot(x_arc, y_arc, c = 'red', lw = 5)
    ax.vlines(x = path_a[0, n], ymin = path_a[1, n] - (r * 1.1), ymax = path_a[1, n] + (r * 1.1), color = 'k', lw = 0.5)
    ax.hlines(y = path_a[1, n], xmin = path_a[0, n] - (r * 1.1), xmax = path_a[0, n] + (r * 1.1), color = 'k', lw = 0.5)

def corner_R_simulator(R, D, z, ap, ae, theta = 38, L_edge = 6, Vc = 90, fz = 0.03, N = 3000, type = 'concentric', initial_phase = 0, path_b_plot = False, n = None, anime = False, dpi = 60):
    start = time.time()
    print('演算開始')
    df_contact_arc = contact_arc_v2(R, D, z, ae, Vc, fz, type, N, n, path_b_plot, anime, dpi)
    print(f'接触弧長さ演算完了{np.round(time.time() - start, decimals = 3)}s')

    r = D / 2
    p = pi * D / z
    T = df_contact_arc['time'].values
    L = df_contact_arc['contact_arc'].values

    T_right = abs(T - (max(T) / 3))
    n_T_right = np.where(T_right == min(T_right))[0][0]
    T_left = abs(T + (max(T) / 3))
    n_T_left = np.where(T_left == min(T_left))[0][0]
    N_anime = int(N / 50)
    N_anime = np.linspace(n_T_left, n_T_right, N_anime, dtype = int)

    N = len(T)
    omega_c = (1e3 * Vc / 60) / r
    alpha = omega_c * T
    alpha = np.diff(alpha)
    alpha_edge_0 = np.linspace(0, 2 * pi, z)
    trans = list(range(1, z))
    trans.append(0)

    S_1 = -abs(ap * tan(radians(theta)))
    S_2 = 0
    S_4 = L.reshape([N, 1])
    S_3 = S_4 - ap * tan(radians(theta))
    n_negativ_edge = int(S_1 / p - 1)
    S_0 = p * n_negativ_edge
    print(f'条件設定完了{np.round(time.time() - start, decimals = 3)}s')

    x_max_0 = L_edge * tan(radians(theta))
    x_edge_0 = np.zeros([z, 10])
    for i in range(n_negativ_edge, z + n_negativ_edge):
        x_edge_0[i - n_negativ_edge] = np.linspace(p * i, x_max_0 + (p * i), 10)
    x_edge_0 -= r * np.radians(initial_phase)
    y_edge = np.linspace(0, L_edge, 10)
    x_edge = np.zeros([N, z, 10])
    x_edge[0] = x_edge_0
    for n in range(1, N):
        x_edge[n] = x_edge[n - 1] - r * sin(alpha[n - 1])
        x_judge = x_edge[n, 0, 0]
        if x_judge < S_0:
            x_edge[n] = x_edge[n, trans]
            x_edge[n, -1] += pi * D
    print(f'座標設定完了{np.round(time.time() - start, decimals = 3)}s')

    P_x = x_edge[:, :, 0]
    P_y = np.zeros([N, z])
    Q_x = P_x + ap * tan(radians(theta))
    Q_y = np.zeros([N, z])
    P_y = np.where((S_1 <= P_x) & (P_x < S_2), abs(P_x) / tan(radians(theta)), P_y)
    P_y = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, P_y)
    Q_x = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, Q_x)
    Q_x = np.where((S_3 <= P_x) & (P_x < S_4), S_4, Q_x)
    Q_y = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, Q_y)
    Q_y = np.where((S_1 <= P_x) & (P_x < S_3), ap, Q_y)
    Q_y = np.where((S_3 <= P_x) & (P_x < S_4), (S_4 - P_x) / tan(radians(theta)), Q_y)
    P_x = np.where((S_1 <= P_x) & (P_x < S_2), 0, P_x)
    P_x = np.where((P_x < S_1) | (S_4 <= P_x), np.nan, P_x)
    print(f'交点演算完了{np.round(time.time() - start, decimals = 3)}s')

    L_contact_edge = sqrt((Q_x - P_x) ** 2 + (Q_y - P_y) ** 2)
    L_contact_edge_sum = np.nansum(L_contact_edge, axis = 1)
    N_contact_edge = np.nan_to_num(L_contact_edge)
    N_contact_edge = np.count_nonzero(N_contact_edge, axis=1)
    ans = pd.DataFrame(
        {
            'time':T,
            'L_contact_arc':L,
            'N_SCE':N_contact_edge,
            'L_SCE':L_contact_edge_sum
        }
    )
    print(f'演算終了{np.round(time.time() - start, decimals = 3)}s')

    if not n:
        n = np.where(L_contact_edge_sum == max(L_contact_edge_sum))[0][0]
    L_n = L[n]
    L_n = np.linspace(0, L_n, 10)
    fig, ax = plt.subplots()
    ax.fill_between(L_n, 0, ap, color = 'deepskyblue', alpha = 0.5)
    ax.set_aspect('equal')
    plot_format(
            ax,
            [-1, math.ceil(1.1 * max(L))],
            [0, math.ceil(1.1 * ap)]
    )
    for z_i in range(z):
        ax.plot(P_x[n, z_i], P_y[n, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
        ax.plot(Q_x[n, z_i], Q_y[n, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
        ax.plot(x_edge[n, z_i], y_edge, c = 'k')

    fig, ax = plt.subplots()
    ax.plot(T, L_contact_edge_sum, c = 'k')
    plot_format(
            ax,
            [T[n_T_left], T[n_T_right]],
            [0, max(L_contact_edge_sum) * 1.1],
            fontsize = 15,
            pad = [5, 3]
    )
    fig, ax = plt.subplots()
    ax.plot(T, N_contact_edge, c = 'k')
    plot_format(
            ax,
            [T[n_T_left], T[n_T_right]],
            [0, max(N_contact_edge) + 1],
            fontsize = 15,
            pad = [5, 3]
    )

    if anime:
        print(f'アニメーション開始{np.round(time.time() - start, decimals = 3)}s')
        fig, ax = plt.subplots()
        plot_format(
                ax,
                [-1, math.ceil(1.1 * max(L))],
                [0, math.ceil(1.1 * ap)]
        )
        frames = []
        for n in N_anime:
            L_n = L[n]
            L_n = np.linspace(0, L_n, 10)
            frame = [ax.fill_between(L_n, 0, ap, color = 'deepskyblue', alpha = 0.5)]
            for z_i in range(z):
                frame += ax.plot(x_edge[n, z_i], y_edge, c = 'k')
                frame += ax.plot(P_x[n, z_i], P_y[n, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
                frame += ax.plot(Q_x[n, z_i], Q_y[n, z_i], marker = 'o', markersize = 7, markeredgecolor = 'k', c = 'r', zorder = 10)
                frames.append(frame)
        anime = ArtistAnimation(fig, frames, interval = 50)
        os.makedirs('simulate', exist_ok = True)
        anime.save('simulate/test.gif', writer = "pillow")

    print(f'終了{np.round(time.time() - start, decimals = 3)}s')

    return ans
