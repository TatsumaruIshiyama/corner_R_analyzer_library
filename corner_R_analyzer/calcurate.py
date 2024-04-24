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
def contact_arc_max(R, D, ae):
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
def contact_arc(R, D, ae, dx = 0.05):
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

    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_xlim(-R - 5, R + 5)
    ax.set_ylim(-R - 5, R + 5)
    ax.plot(x_work, y_a, color = 'black')
    ax.plot(x_tool, y_tool, color = 'gray', linestyle = 'dashdot')

    df_work = pd.DataFrame(data = {'y_a':y_a, 'y_b':y_b}, index = x_work)
    df_tool = pd.DataFrame(data = {'y_tool':y_tool}, index = x_tool)
    df = pd.concat([df_work, df_tool], axis = 1)

    frames = []
    contact_arc = []
    ae_true = []

    for i in range(len(x_tool)):
        x_endmill = np.arange(-r, r + dx, dx)
        y_endmill = np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        y_endmill_fig = -np.sqrt(r ** 2 - (x_endmill) ** 2) + y_tool[i]
        x_endmill = np.round(x_endmill + x_tool[i], decimals = 3)
        frame_1 = ax.plot(x_endmill, y_endmill, color = 'blue')
        frame_2 = ax.plot(x_endmill, y_endmill_fig, color = 'blue')

        df_endmill = pd.DataFrame(data = {'x_endmill':x_endmill, 'y_endmill':y_endmill}, index = x_endmill)
        df_calc = pd.concat([df, df_endmill], axis = 1)
        df_calc['L_a'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_a']) ** 2)
        df_calc['L_b'] = np.sqrt((df_calc['x_endmill'] - df_calc.index.values) ** 2 + (df_calc['y_endmill'] - df_calc['y_b']) ** 2)
        P_x = df_calc['L_a'].idxmin()

        df_cut = df_calc.loc[P_x:]
        Q_x = df_cut['L_b'].idxmin()

        if not P_x == 'nan':
            P_y = df_calc.loc[P_x, 'y_a']
            Q_y = df_calc.loc[Q_x, 'y_b']
            
            vec_P = [P_x - x_tool[i], P_y - y_tool[i]]
            vec_Q = [Q_x - x_tool[i], Q_y - y_tool[i]]
            inner = np.inner(vec_P, vec_Q)
            theta = np.arccos(inner / (np.linalg.norm(vec_P) * np.linalg.norm(vec_Q)))
            contact_arc.append([x_tool[i], r * theta])
            ae_true.append([x_tool[i], r * (1 - np.cos(theta))])
            
            # frame_5 = ax.plot(df_calc.loc[Q_x:].index, df_calc.loc[Q_x:, 'y_b'], color = 'gray')
            # frame_3 = ax.plot(P_x, P_y, marker = '*', color = 'red', markersize = 10)
            # frame_4 = ax.plot(Q_x, Q_y, marker = '*', color = 'red', markersize = 10)

            # frames.append(frame_1 + frame_2 + frame_3 + frame_4 + frame_5)

        else:
            frames.append(frame_1 + frame_2)

    # anime = ArtistAnimation(fig, frames, interval = dx * 10)
    # plt.show()
    # anime.save('animation.gif', writer='pillow')

    contact_arc = pd.DataFrame(contact_arc, columns = ['tool_position', 'L'])
    fig, ax = plt.subplots()
    ax.plot(contact_arc['tool_position'], contact_arc['L'], color = 'black')
    ax.set_xlim(min(contact_arc['tool_position']), max(contact_arc['tool_position']))
    ax.set_ylim(0, max(contact_arc['L']) + 1)
    ax.plot(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'], max(contact_arc['L']), marker = '.', markersize = 15, color = 'black')
    ax.text(contact_arc.loc[contact_arc['L'].idxmax(), 'tool_position'] + 0.1, max(contact_arc['L']), str(np.round(max(contact_arc['L']), decimals = 2)))

    ae_true = pd.DataFrame(ae_true, columns = ['tool_position', 'ae_true'])
    fig, ax = plt.subplots()
    ax.plot(ae_true['tool_position'], ae_true['ae_true'], color = 'black')
    ax.set_xlim(min(ae_true['tool_position']), max(ae_true['tool_position']))
    ax.set_ylim(0, max(ae_true['ae_true']) + 0.1)
    ax.plot(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'], max(ae_true['ae_true']), marker = '.', markersize = 15, color = 'black')
    ax.text(ae_true.loc[ae_true['ae_true'].idxmax(), 'tool_position'] + 0.1, max(ae_true['ae_true']), str(np.round(max(ae_true['ae_true']), decimals = 2)))

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
def ae_from_L(R, D, L, ae_lim = 1):
    L = np.matrix(L).T
    all_ae = np.arange(1 / 10000, ae_lim + 1 / 10000, 1 / 10000, dtype = 'float32')
    all_L = contact_arc_max(R, D, all_ae)
    all_ae = np.matrix([all_ae] * len(L))
    all_L = np.matrix([all_L] * len(L))
    L_diff = np.abs(all_L - L)
    ae = all_ae[np.where(L_diff == np.min((L_diff), axis = 1))]
    L_diff = L_diff[np.where(L_diff == np.min((L_diff), axis = 1))]
    ae = np.array(ae[np.where(L_diff <= 1 / 100)])[0]
    return ae
# %%
def cut_depth(R, D, theta, z_lim = 10, ae_lim = 1, check = False):
    theta_rad = radians(theta)
    ans = []
    for z in range(2, z_lim + 1):

        p = pi * D / z
        ap = np.round(p / tan(theta_rad), decimals = 3)

        L_min = p
        L_max = p * int(z / 2) + p
        L = np.arange(L_min, L_max, p)
        ae = ae_from_L(R, D, L, ae_lim)
        if len(ae):
            ae = np.round(min(ae), decimals = 3)
            ans.append([z, ap, ae])
            if check:
                contact_area(R, D, z, ap, ae, theta = theta, dx = 0.05)
    ans = pd.DataFrame(ans, columns = ['z', 'Ap', 'Ae'])
    ans = ans.set_index('z')
    return ans
# %%
