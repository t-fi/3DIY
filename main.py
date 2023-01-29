import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

WIN_NAME_FULL = "full"
WIN_NAME_ZOOM = "zoom"

x0 = -1
y0 = -1

fig, ax = plt.subplots(figsize=(5.5, 5.5))


def analyze_cutout(cutout):
    ax.set_aspect(1.)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    hist_x = np.sum(cutout, axis=0)
    hist_y = np.sum(cutout, axis=1)

    ax.matshow(cutout)
    ax_histx.bar(range(len(hist_x)), hist_x)
    ax_histy.barh(range(len(hist_y)), hist_y)

    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)


def choose_start(event, x, y, flags, param):
    global x0, y0
    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0 = x, y


def main():
    video_path = Path("/home/oba/data/crop.mp4")
    if not video_path.exists():
        raise ValueError("The video file does not exist: " + str(video_path))

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError("Could not open video :(")

    cv2.namedWindow(WIN_NAME_FULL)

    ret, frame = video.read()
    cv2.imshow(WIN_NAME_FULL, frame)
    cv2.setMouseCallback(WIN_NAME_FULL, choose_start)

    while cv2.waitKey(20):
        if x0 != -1:
            print(f"chose start at: {[x0, y0]}")
            break

    cv2.setMouseCallback(WIN_NAME_FULL, lambda *args: None)
    cv2.namedWindow(WIN_NAME_ZOOM)

    i = 0
    while cv2.waitKey(0):
        ret, frame = video.read()
        if not ret:
            return

        print(f"created figure {i}")
        i += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cutout_radius = 15
        cutout = gray[y0 - cutout_radius: y0 + cutout_radius, x0 - cutout_radius: x0 + cutout_radius]

        cv2.imshow(WIN_NAME_FULL, frame)
        cv2.imshow(WIN_NAME_ZOOM, analyze_cutout(cutout))


if __name__ == '__main__':
    main()
