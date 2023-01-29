import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("data.json", "r") as f:
        positions = json.load(f)

    m = len(positions) / 17.1
    tm = [t % m for t in range(len(positions))]
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])

    fig, ax = plt.subplots()
    ax.scatter(tm, x, c="r")
    ax.set_ylabel("x")
    ax.set_xlabel("frame number")
    ax2 = ax.twinx()
    ax2.scatter(tm, y, c="g")
    ax2.set_ylabel("y")
    plt.title("Positions over time")
    plt.show()


if __name__ == '__main__':
    main()
