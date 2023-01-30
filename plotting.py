import json
import matplotlib.pyplot as plt
import numpy as np


def rotate(x, y, theta):
    c = np.cos(theta)
    s = np.sin(theta)

    x[:] = x * c - y * s
    y[:] = x * s + y * c


def main():
    with open("data.json", "r") as f:
        positions = json.load(f)

    m = len(positions) / 50.0185
    tm = [t % m for t in range(len(positions))]
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])

    rotate(x, y, 0.016)

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
        ax.scatter(tm, x, c="r")
        ax.set_ylabel("x")
        ax.set_xlabel("frame number")
        ax2 = ax.twinx()
        ax2.scatter(tm, y, c="g")
        ax2.set_ylabel("y")
        plt.title("Positions over time")
        plt.show()

    if False:
        plt.scatter(x, y)
        plt.show()


if __name__ == '__main__':
    main()
