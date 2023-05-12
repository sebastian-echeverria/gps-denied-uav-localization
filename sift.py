
import os
from pathlib import Path

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from pix2coords import ImageProjector

def find_matching_points(img1, img2):
    """Returns two sets of matching points between the two given images."""

    # Find the keypoints and descriptors with SIFT
    print("Finding keypoints with SIFT", flush=True)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find matches using FLANN.
    print("Finding matches using FLANN", flush=True)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 8
    if len(good_matches) < MIN_MATCH_COUNT:
        raise Exception("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
    else:
        print(f"Found {len(good_matches)} matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


def show_matches(img1, img2, dst, kp1, kp2, good, matchesMask, output_file, block_plot=False, out_color='gray'):
    # Draw the projection of the first image into the second.
    poly_color = 0
    if out_color != 'gray':
        poly_color = (255, 0, 0)
    img3 = cv.polylines(img2, [np.int32(dst)], True, poly_color, 3, cv.LINE_AA)

    # Add the keypoint matches between the images.
    draw_params = dict(matchColor=(0, 255, 0),   # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    if kp1 is not None and kp2 is not None and good is not None:
        img3 = cv.drawMatches(img1, kp1, img3, kp2, good, None, **draw_params)

    plt.clf()
    plt.imshow(img3, out_color)

    # Save to file
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)

    if block_plot:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)

def align_and_show(template_path: str, input_path: str, show_plot: bool=True):
    cv_color = 0
    mode = "detect"
    block_plot = False
    out_color = 'gray'

    print(f"Finding image: {template_path}")
    template = cv.imread(template_path, cv_color)
    mosaic = cv.imread(input_path, cv_color)

    if mode == "detect":
        try:
            src_pts, dst_pts, kp1, kp2, good_matches = find_matching_points(template, mosaic)
        except Exception as ex:
            print(f"Error: {str(ex)}")
            return

    # Get GPS from set of matching points.
    projector = ImageProjector()
    projector.load_template_image(template_path)
    projector.load_input_image(input_path)
    matchesMask = projector.calculate_homography(src_pts, dst_pts)
    gps_coords, projected_corners, _ = projector.infer_coordinates()
    print(f"Coords: {gps_coords}")

    # Show matches, as well as KPs
    if show_plot:
        show_matches(template, mosaic, projected_corners, kp1, kp2, good_matches, matchesMask, "data/matching1.png", block_plot, out_color)

    return projector, gps_coords
