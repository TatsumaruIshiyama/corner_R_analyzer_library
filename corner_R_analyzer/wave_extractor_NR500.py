#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import copy
# %%
# %%
def bar_format(ax, lim, fontsize = 15, flame_width = 1.5, scale_length = 5, pad = {0, 0}, grid_width = 0.5):
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
        width = 0,
        length = 0
        )
    ax.set_xlim(lim[0][0], lim[0][1])
    ax.set_ylim(lim[1][0], lim[1][1])
    ax.grid(color = 'black', linewidth = grid_width)
# %%
def read_standard(path, col_name, col_st):
    n_data = np.loadtxt(
        path,
        delimiter = ',',
        skiprows = 10,
        max_rows = 1,
        usecols = 1,
        dtype = int
    )
    n_st_col = col_name.index(col_st)
    n_st_col += 2
    
    df_st = np.loadtxt(
        path,
        delimiter = ',',
        skiprows = 70,
        max_rows = n_data + 70 - 1 - 3,
        usecols = n_st_col,
        dtype = float
    )
    return df_st
#%%
def binaly(data, threshold, b):
    data_bin = np.where(data < threshold, 0, data)
    data_bin = np.where(threshold <= data, 1, data_bin)
    data_bin = np.convolve(data_bin, b, mode = 'same')
    data_bin = np.where(0.01 <= data_bin, 1, data_bin)
    data_bin = data_bin.astype(dtype = 'int')
    plt.plot(data_bin)
    return data_bin
def binaly_dinamo(data, threshold, b, n_level):
    data_bin = np.where(data < threshold, 0, data)
    data_bin = np.where(threshold <= data, 1, data_bin)
    i = 0
    while i <= n_level:
        data_bin = np.convolve(data_bin, b, mode = 'same')
        data_bin = np.where(0.01 <= data_bin, 1, data_bin)
        i += 1
    if data_bin[-1] == 1:
        data_bin[-1] = 0
    data_bin = data_bin.astype(dtype = 'int')

    return data_bin
#%%
def check(data_st, threshold, skip, n_conv):
    n_data = len(data_st)
    n_conv = int(n_data / n_conv)
    b = np.ones(n_conv) / n_conv
    data_calc = data_st.copy()
    data_calc = abs(data_calc)
    threshold *= max(data_calc)

    data_bin_R = np.flip(data_calc)
    data_bin_L = binaly(data_calc, threshold, b)
    data_bin_R = binaly(data_bin_R, threshold, b)

    id_ex_L = np.diff(data_bin_L)
    id_ex_L = np.where(id_ex_L == 1)[0]
    id_ex_R = np.diff(data_bin_R)
    id_ex_R = np.where(id_ex_R == 1)[0]
    id_ex_R = n_data - id_ex_R
    id_ex_R = np.flip(id_ex_R)
    len_id = id_ex_R - id_ex_L
    len_max = max(len_id)
    id_ex_L = np.delete(id_ex_L, np.where(len_id < len_max * skip))
    id_ex_R = np.delete(id_ex_R, np.where(len_id < len_max * skip))
    fig, ax = plt.subplots()
    ax.plot(
        np.linspace(1, len(data_st), len(data_st)),
        data_st,
        c = 'black'
    )
    ax.hlines(
        y = threshold,
        xmin = 0,
        xmax = len(data_st),
        color = 'red'
    )
    for i in range(len(id_ex_L)):
        ax.plot(list(range(id_ex_L[i], id_ex_R[i])), data_st[id_ex_L[i]:id_ex_R[i]])
    return id_ex_L, id_ex_R
def check_dinamo(data, fs, threshold = 4, skip = 0.7, n_conv = 500, n_level = 10, t_standard = 0.1):
    n_data = len(data)
    n_conv = int(n_data / n_conv)
    b = np.ones(n_conv) / n_conv
    data_calc = data.copy()
    data_calc = abs(data_calc)
    threshold = max(data_calc[:int(t_standard * fs)]) * threshold

    data_bin = binaly_dinamo(data_calc, threshold, b, n_level)

    id_ex_L = np.diff(data_bin)
    id_ex_R = np.where(id_ex_L == -1)[0]
    id_ex_L = np.where(id_ex_L == 1)[0]
    len_id = id_ex_R - id_ex_L
    len_max = max(len_id)
    id_ex_L = np.delete(id_ex_L, np.where(len_id < len_max * skip))
    id_ex_R = np.delete(id_ex_R, np.where(len_id < len_max * skip))
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(data, c = 'orange')
    ax2.plot(data_bin, c = 'blue')
    for i in range(len(id_ex_L)):
        ax.plot(
            list(range(id_ex_L[i], id_ex_R[i])),
            data[id_ex_L[i]:id_ex_R[i]],
            c = 'red'
    )
    return id_ex_L, id_ex_R

#%%
def extract_df(path, col_name, id_ex_L, id_ex_R):
    dfs_ex = []
    for i in range(len(id_ex_L)):
        df_ex = np.loadtxt(
            path,
            delimiter = ',',
            skiprows = id_ex_L[i] + 70,
            max_rows = id_ex_R[i] - id_ex_L[i] + 70 - 1,
            usecols = list(range(2, 2 + len(col_name))),
            dtype = float            
        )
        dfs_ex.append(df_ex)
    if len(dfs_ex) == 1:
        dfs_ex = dfs_ex[0]
    return dfs_ex
#%%
def FFT(data, samplerate):
    N = len(data)
    F = np.fft.fft(data, norm = 'ortho')
    amp = np.abs(F)
    amp = amp[1:int(N / 2)]
    freq = np.fft.fftfreq(N, d = 1 / samplerate)
    freq = freq / 1e3
    freq = freq[1:int(N / 2)]
    return freq, amp
# %%
def filter(data, sample_rate, type, fp, fs, gpass = 3, gstop = 40):
    if type == 'band':
        fp = np.array(fp)
        fs = np.array(fs)

    fn = sample_rate / 2

    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, type)

    data_filt = signal.filtfilt(b, a, data)    
    return data_filt
# %%
def plot_format(ax, xlim, ylim, fontsize = 15, flame_width = 1.5, scale_length = 5, pad = [0, 0], grid_width = 0.5):
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
def show_wave(data, sampling_rate, axis):
    time = data.index * (1 / sampling_rate)
    xlim = time[-1]
    for i in range(len(axis)):
        data_plot = data[axis[i]]
        ylim = max(abs(data_plot)) * 1.1
        fig, ax = plt.subplots()   
        ax.plot(
            time,
            data_plot,
            c = 'black'
        )
        plot_format(
            ax,
            xlim = [0, xlim],
            ylim = [-ylim, ylim]
        )
        ax.set_title(axis[i])
# %%
def show_peak(ax, freq, amp, sampling_rate, sensitivity, ymax):
    text = np.array(np.round(freq, decimals = 2), dtype = '<U')
    peak_id = signal.find_peaks(amp, distance = sampling_rate / sensitivity)[0]
    ax.plot(freq, amp, color = 'black')
    ax.plot(freq[peak_id], amp[peak_id], marker = 'o', color = 'red', linewidth = 0)
    for i in range(len(peak_id)):
        ax.text(s = text[peak_id[i]], x = freq[peak_id[i]], y = amp[peak_id[i]] + ymax / 30)
# %%
def show_FFT(data, sampling_rate, axis, peak, sensitivity):
    for i in range(len(axis)):
        data_fft = data[axis[i]]
        freq, amp = FFT(data_fft, sampling_rate)
        ylim = max(amp) * 1.1
        fig, ax = plt.subplots(figsize = (8, 3))
        ax.plot(
            freq,
            amp,
            c = 'k'
        )
        plot_format(
            ax,
            xlim = [0, int(sampling_rate / 2e3)],
            ylim = [0, ylim],
            pad = [3, 3]
        )
        ax.set_title(axis[i])
        if peak:
            show_peak(ax, freq, amp, sampling_rate, sensitivity, ylim)
# %%
def show_filter(data, data_filt, sampling_rate, axis):
    time = (data.index / sampling_rate)
    xlim = time[-1]
    ylim = max(abs(data[axis])) * 1.1
    fig, ax = plt.subplots()
    ax.plot(time, data[axis], c = 'k')
    plot_format(
    ax,
    xlim = [0, xlim],
    ylim = [-ylim, ylim],
    pad = [3, 3]
    )
    ax.set_title('origin')
    fig, ax = plt.subplots()
    ax.plot(time, data_filt, c = 'k')
    plot_format(
    ax,
    xlim = [0, xlim],
    ylim = [-ylim, ylim],
    pad = [3, 3]
    )
    ax.set_title('filter')
    freq, amp = FFT(data[axis], sampling_rate)
    ylim = max(amp) * 1.1
    fig, ax = plt.subplots(figsize = (8, 3))
    ax.plot(
        freq,
        amp,
        c = 'k'
    )
    plot_format(
        ax,
        xlim = [0, int(sampling_rate / 2e3)],
        ylim = [0, ylim],
        pad = [3, 3]
    )
    ax.set_title('FFT origin')
    freq_filt, amp_filt = FFT(data_filt, sampling_rate)
    fig, ax = plt.subplots(figsize = (8, 3))
    ax.plot(
        freq_filt,
        amp_filt,
        c = 'k'
    )
    plot_format(
        ax,
        xlim = [0, int(sampling_rate / 2e3)],
        ylim = [0, ylim],
        pad = [3, 3]
    )
    ax.set_title('FFT filt')