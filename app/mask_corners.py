import cv2 as cv
import numpy as np


# Functions to find corners of a polygon in a given mask.
def find_corners(mask):
    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #print(contours)
    return _getQuadrangleWithRegularOrder(contours[0])


def _getApprox(contour, alpha):
    epsilon = alpha * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    return approx


# find appropriate epsilon
def _getQuadrangle(contour):
    alpha = 0.1
    beta = 2 # larger than 1
    approx = _getApprox(contour, alpha)
    if len(approx) < 4:
        while len(approx) < 4:
            alpha = alpha / beta
            approx = _getApprox(contour, alpha)  
        alpha_lower = alpha
        alpha_upper = alpha * beta
    elif len(approx) > 4:
        while len(approx) > 4:
            alpha = alpha * beta
            approx = _getApprox(contour, alpha)  
        alpha_lower = alpha / beta
        alpha_upper = alpha
    if len(approx) == 4:
        return approx
    alpha_middle = (alpha_lower * alpha_upper ) ** 0.5
    approx_middle = _getApprox(contour, alpha_middle)
    while len(approx_middle) != 4:
        if len(approx_middle) < 4:
            alpha_upper = alpha_middle
            approx_upper = approx_middle
        if len(approx_middle) > 4:
            alpha_lower = alpha_middle
            approx_lower = approx_middle
        alpha_middle = ( alpha_lower * alpha_upper ) ** 0.5
        approx_middle = _getApprox(contour, alpha_middle)
    return approx_middle


def _getQuadrangleWithRegularOrder(contour):
    approx = _getQuadrangle(contour)
    hashable_approx = [tuple(a[0]) for a in approx]
    sorted_by_axis0 = sorted(hashable_approx, key=lambda x: x[0])
    sorted_by_axis1 = sorted(hashable_approx, key=lambda x: x[1])
    topleft_set = set(sorted_by_axis0[:2]) & set(sorted_by_axis1[:2])
    assert len(topleft_set) == 1
    topleft = topleft_set.pop()
    topleft_idx = hashable_approx.index(topleft)
    approx_with_reguler_order = [ approx[(topleft_idx + i) % 4] for i in range(4) ]
    return approx_with_reguler_order
