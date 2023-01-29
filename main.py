import cv2
import numpy as np
from pathlib import Path

WIN_NAME_FULL = "full"
WIN_NAME_ZOOM = "zoom"
CUTOUT_RADIUS = 40

x0 = -1
y0 = -1

positions = []


def analyze_cutout(cutout):
    global x0, y0

    ret, thr = cv2.threshold(cutout, 40, 255, cv2.THRESH_BINARY)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thr)

    if ret == 1:
        raise ValueError("Only the background label was found :(")

    largest_component_index = np.argmax(stats[1:, 4]) + 1

    blob_center = centroids[largest_component_index].astype(int)

    x0 += blob_center[0] - CUTOUT_RADIUS
    y0 += blob_center[1] - CUTOUT_RADIUS

    visualization = np.hstack((cutout, thr))

    return cv2.resize(visualization, (0, 0), fx=4, fy=4)


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

    while cv2.waitKey(0):
        ret, frame = video.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cutout = gray[y0 - CUTOUT_RADIUS: y0 + CUTOUT_RADIUS, x0 - CUTOUT_RADIUS: x0 + CUTOUT_RADIUS]
        cv2.circle(frame, (x0, y0), 10, (255, 0, 255), 2)

        cv2.imshow(WIN_NAME_ZOOM, analyze_cutout(cutout))
        cv2.circle(frame, (x0, y0), 10, (0, 255, 255), 2)

        positions.append((x0, y0))

        big_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        cv2.imshow(WIN_NAME_FULL, big_frame)




if __name__ == '__main__':
    main()
