import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from scipy import ndimage
from scipy import signal


def schmoof(t, pos, num_vals, window_size):
    t_base = np.linspace(min(t), max(t), num_vals)
    smooth = np.empty_like(t_base)
    for i in trange(len(t_base)):
        t_local = t_base[i]
        selected_idxs = np.where(np.abs(t - t_local) < window_size)
        selected_ts = t[selected_idxs]

        weights = 1 / (1 + (np.abs(t_local - selected_ts)))
        weights /= sum(weights)

        smooth[i] = np.sum(pos[selected_idxs] * weights)

    return t_base, smooth


def rotate(x, y, theta):
    c = np.cos(theta)
    s = np.sin(theta)

    x[:] = x * c - y * s
    y[:] = x * s + y * c


def main():
    with open("data.json", "r") as f:
        positions = json.load(f)

    m = len(positions) / 50.0185
    tm = np.array([t % m for t in range(len(positions))])

    sorted_idxs = np.argsort(tm)
    t = tm[sorted_idxs]
    x = np.array([p[0] for p in positions])[sorted_idxs]
    y = np.array([p[1] for p in positions])[sorted_idxs]

    rotate(x, y, 0.016)

    t_smooth, x_smooth = schmoof(t, x, 2000, 1)
    x_smooth = ndimage.gaussian_filter1d(x_smooth, 5)

    peaks = t_smooth[signal.find_peaks(x_smooth)[0]]

    valleys = t_smooth[signal.find_peaks(-x_smooth)[0]]

    if False:
        fig, ax = plt.subplots()
        ax.plot(x, c="r")
        ax.set_ylabel("x")
        ax.set_xlabel("frame number")
        ax2 = ax.twinx()
        ax2.plot(y, c="g")
        ax2.set_ylabel("y")
        plt.title("Positions over time")
        plt.show()

    if True:
        fig, ax = plt.subplots()
        for peak in peaks:
            ax.axvline(peak, c="r")
        for valley in valleys:
            ax.axvline(valley, c="b")
        ax.scatter(t, x, c="r")
        ax.plot(t_smooth, x_smooth, c="b")
        ax.set_ylabel("x")
        ax.set_xlabel("frame number")
        ax2 = ax.twinx()
        ax2.scatter(t, y, c="g")
        ax2.set_ylabel("y")
        plt.title("Positions over time")
        plt.show()

    if True:
        plt.scatter(x, y)
        plt.show()

    if True:
        deltas = np.diff(sorted(np.hstack((peaks, valleys))))
        plt.hist(deltas)
        plt.show()
        mean = np.mean(deltas)
        std = np.std(deltas)
        print(f"frame deltas are roughly: {mean} +- {std}")
        print(f"at 240fps, in hz: {240/mean} +- {240/std}")


if __name__ == '__main__':
    main()
