
import numpy as np
import cv2 as cv

from osgeo import osr
from osgeo import gdal

from utils import printd


class CoordinateConversor():
    """Converts coordinates from a GeoTIFF base image."""

    def __init__(self, mosaic_gdal):
        # this code takes 0.5ms, not much compared to the rest
        # if needed, it could be refactored to be done only once
        self.c, self.a, self.b, self.f, self.d, self.e = mosaic_gdal.GetGeoTransform()

        projection_name = mosaic_gdal.GetProjection()
        if projection_name == "":
            raise RuntimeError("Can't initiate conversor, image does not have GeoTIFF projection info.")

        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_name)
        srsLatLong = srs.CloneGeogCS()
        self.coord_transform = osr.CoordinateTransformation(srs, srsLatLong)

    def pixel_to_coord(self, col, row):
        """Returns global coordinates to pixel center using base-0 raster index"""
        xp = self.a * col + self.b * row + self.c
        yp = self.d * col + self.e * row + self.f
        coords = self.coord_transform.TransformPoint(xp, yp, 0)
        return coords


class ImageProjector():

    def __init__(self) -> None:
        self.template_image = None  # An OpenCV loaded image.
        self.input_image = None     # A GDAL loaded image.
        self.homography = None      # A matrix to project from template to input.

    def load_template_image(self, template_image_path: str) -> None:
        self.template_image = cv.imread(template_image_path, 0)

    def load_input_image(self, input_image_path: str) -> None:        
        self.input_image = gdal.Open(input_image_path)
        printd(f"Input image {input_image_path} loaded as GDAL")

    def calculate_homography(self, src_pts, dst_pts) -> list:
        """Calculates and stores the homography given two sets of points, and returns the matches mask."""
        self.homography, matches_mask = _find_homography(src_pts, dst_pts)
        return matches_mask

    def project_template(self):
        """Projects the template using the homography, and returns the corners of the projected image in the input image."""
        if self.template_image is None:
            raise RuntimeError("No template image has been set.")
        if self.homography is None:
            raise RuntimeError("Homography has not been calculated or set.")

        return _project_image(self.template_image, self.homography)
    
    def infer_coordinates(self):
        """Given the an image and the reference mosaic, plus a homography transformation, it retuns the the GPS
        coordinates the centroid of the template image, its corners in pixels, plus whether the shape looks rectangular-like or not."""
        if self.input_image is None:
            raise RuntimeError("No input image has been set.")
        
        projected_corners = self.project_template()
        printd(f"Projection: {projected_corners}")

        diagonals_intersection = _calculate_diagonals_intersection(projected_corners)
        printd(f"Diagonals Intersection: {diagonals_intersection}")
        has_good_shape = _check_if_rectangular_like(projected_corners, diagonals_intersection)

        conversor = CoordinateConversor(self.input_image)
        gps_coords = conversor.pixel_to_coord(diagonals_intersection[0], diagonals_intersection[1])
        printd(f"GPS coords: {gps_coords}")

        return gps_coords, projected_corners, has_good_shape    


def _find_homography(src_pts: list, dst_pts: list):
    """Calculates the homography matrix from the matching points."""
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return M, matchesMask


def _project_image(img1, homography):
    """Projects the image using the homography matrix."""
    printd(f"Image shape: {img1.shape}")
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, homography)
    return dst


def _calculate_diagonals_intersection(points):
    """Calculates the intersection of the diagonals of the quadrillateral defined by the given points."""
    line1 = (points[0][0], points[2][0])
    line2 = (points[1][0], points[3][0])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def _check_if_rectangular_like(pts, centroid):
    """Checks if the points form a rectangle-ish shape."""
    # First calculate distances between every corner and centroid.
    dists = np.zeros(4)
    for i in range(0, 4):
        dists[i] = np.linalg.norm(pts[i] - centroid)
    printd(f"Distances: {dists}")

    # Now calculate percentual differences between the previous distances. If there is a percentual difference
    # higher than the threshold, it means the projections has a non-right-angle-ish corner.
    ANGLE_PERCENT_THRESHOLD = 20
    bad_shape = False
    diffsp = np.zeros(4)
    for i in range(0, 4):
        diffsp[i] = abs(dists[i] - dists[(i+1)%4])/dists[i]*100
        if diffsp[i] > ANGLE_PERCENT_THRESHOLD:
            bad_shape = True
    printd(f"% Diffs: {diffsp}")

    if bad_shape:
        printd("BAD SHAPE!!!!!!")
        return False
    else:
        return True
