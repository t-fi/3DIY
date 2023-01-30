import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

WIN_NAME_FULL = "full"
WIN_NAME_ZOOM = "zoom"
CUTOUT_X = 20
CUTOUT_Y = 50

x0 = 106
y0 = 438

positions = []


def read_frames_from_folder(folder_path: Path, start_frame: int, end_frame: int = None):
    image_files = sorted(folder_path.glob("*.png"))
    images = []
    for file in tqdm(image_files[start_frame:end_frame]):
        images.append(cv2.imread(str(file)))
    return images


def analyze_cutout(cutout):
    global x0, y0

    ret, thr = cv2.threshold(cutout, 40, 255, cv2.THRESH_BINARY)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thr)

    if ret == 1:
        raise ValueError("Only the background label was found :(")

    # width minus height should be small
    roundnesses = stats[1:, 2] - stats[1:, 2] + 4

    metric = stats[1:, 4] / roundnesses

    largest_component_index = np.argmax(metric) + 1

    blob_center = centroids[largest_component_index]

    x0 = int(x0) + blob_center[0] - CUTOUT_X
    y0 = int(y0) + blob_center[1] - CUTOUT_Y

    visualization = np.hstack((cutout, thr))

    return cv2.resize(visualization, (0, 0), fx=4, fy=4)


def choose_start(event, x, y, flags, param):
    global x0, y0
    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0 = x, y


def main():
    image_folder_path = Path("/home/oba/data/frames")
    if not image_folder_path.exists():
        raise ValueError("The image folder does not exist: " + str(image_folder_path))

    images = read_frames_from_folder(image_folder_path, 1570, 3578)

    cv2.imshow(WIN_NAME_FULL, images[0])
    cv2.setMouseCallback(WIN_NAME_FULL, choose_start)

    while cv2.waitKey(20):
        if x0 != -1:
            print(f"chose start at: {[x0, y0]}")
            break

    cv2.setMouseCallback(WIN_NAME_FULL, lambda *args: None)
    cv2.namedWindow(WIN_NAME_ZOOM)

    for frame in tqdm(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cutout = gray[int(y0) - CUTOUT_Y: int(y0) + CUTOUT_Y, int(x0) - CUTOUT_X: int(x0) + CUTOUT_X]
        cv2.circle(frame, (int(x0), int(y0)), 10, (255, 0, 255), 2)

        visualization = analyze_cutout(cutout)
        cutout_centered = gray[int(y0) - CUTOUT_Y: int(y0) + CUTOUT_Y, int(x0) - CUTOUT_X: int(x0) + CUTOUT_X]
        visualization = np.hstack((cv2.resize(cutout_centered, (0, 0), fx=4, fy=4), visualization))
        cv2.imshow(WIN_NAME_ZOOM, visualization)
        cv2.circle(frame, (int(x0), int(y0)), 10, (0, 255, 255), 2)

        positions.append([x0, y0])

        big_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        cv2.imshow(WIN_NAME_FULL, big_frame)
        cv2.waitKey(1)

    with open("data.json", "w") as f:
        f.write(json.dumps(positions, indent=4))


if __name__ == '__main__':
    main()
