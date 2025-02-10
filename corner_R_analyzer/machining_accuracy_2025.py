# %%
import numpy as np
from numpy import sin, cos, tan, sqrt, arcsin, arccos, arctan
import glob
import pandas as pd
from matplotlib import pyplot as plt
import os
def plot_format(ax, xlim, ylim, fontsize = 15, flame_width = 1.5, scale_length = 5, pad = (0, 0), grid_width = 0.5, grid_which = 'major', minor_grid_axis = 'both'):
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
    ax.grid(color = 'black', linewidth = grid_width, which = 'major')
    if grid_which == 'both':
        ax.grid(color = 'black', linewidth = grid_width * 0.6, which = 'minor', axis = minor_grid_axis)
def plot_r_mean(ax, data, z_mes, label = None, size = 5, marker = 'o', markercolor = 'k'):
    ax.invert_yaxis()
    ax.plot(
            data,
            z_mes,
            color = 'white',
            linewidth = 5
            )
    ax.plot(
        data,
        z_mes,
        color = 'black',
        linewidth = 1
        )
    ax.plot(
        data,
        z_mes,
        color = 'black',
        marker = marker,
        markersize = size,
        markerfacecolor = markercolor,
        linewidth = 0,
        label = label
        )
def r_mean(R, condition, nf, xlim, ylim, save = True, cd = r'C:/Users/tatsu/マイドライブ/laboratory/corner_R/analyze_2025/', costum_name = None):
    nf_str = str(nf)
    if len(nf_str) == 1:
        nf_str = '0' + nf_str

    save_dir = cd + rf'analyze_data/R{R}/z{nf_str}/{condition}'
    os.makedirs(save_dir, exist_ok = True)

    File_path = glob.glob(cd + f'data/machining_accuracy/R{R}/{condition}/z{nf_str}/*.csv')
    cut_depth = pd.read_csv(cd + f'data/condition/R{R}.csv')
    cut_depth = cut_depth.loc[cut_depth['nf'] == nf]

    n_exp = len(File_path)
    if not n_exp == 0:
        Ap = cut_depth.loc[:, 'Ap'].values[0]
        Ae = cut_depth.loc[:, 'Ae'].values[0]

        Data = []
        for file_path in File_path:
            Data.append(
                pd.read_csv(file_path)
            )
            
        z_mes = Data[0].loc[:, 'z'].values

        r = []
        for i in range(n_exp):
            r.append(Data[i].loc[:, 'r'].values)
        r = pd.DataFrame(r)
        r_mean = r.mean()
        df_r_mean= pd.DataFrame({'z':z_mes, 'r':r_mean})
        df_r_mean.to_csv(save_dir + '/r_mean.csv', index = None)
        print('CSV saved')

        for i in range(n_exp):
            fig, ax = plt.subplots()
            plot_r_mean(ax, r.iloc[i].values, z_mes, size = 7)
            plot_format(ax, xlim = xlim, ylim = ylim, pad = (6, 3))
            if save:
                if costum_name:
                    plt.savefig(save_dir + f'/{costum_name}_n{i + 1}.png', dpi = 400)
                else:
                    plt.savefig(save_dir + f'/dataplot_n{i + 1}.png', dpi = 400, bbox_inches = 'tight')
                print(f'{i + 1}/{n_exp} DataPlot saved' )
            plt.clf()
            plt.close()

        fig, ax = plt.subplots()  
        plot_r_mean(ax, r_mean.values, z_mes, size = 7)
        plot_format(ax, xlim = xlim, ylim = ylim, pad = (6, 3))
        if save:
            if costum_name:
                plt.savefig(save_dir + f'/{costum_name}_mean.png', dpi = 400)
            else:
                plt.savefig(save_dir + f'/meanplot.png', dpi = 400, bbox_inches = 'tight')
            print('MeanPlot saved')
        plt.clf()
        plt.close()    
        print('Complete all')
        
        return r_mean