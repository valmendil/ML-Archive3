#!/usr/bin/env python3

"""
This file contains all the required methods for the street prediction utilizing
the Hough transform.
"""

import numpy as np
import scipy.ndimage as ndi

from skimage.draw import polygon
from skimage.transform import hough_line


def draw_roads(roads, shape):
    """
    Creates an image with roads drawn as full lines.

    Parameters:
        roads -- ndarray describing all roads to be drawn
        shape -- shape (size) of image

    The parameters are exactly what is returned by find_roads (see there).

    Returns:
        An numpy.ndarray with shape 'shape' and floating point type, where
        background has probability 0 and roads have been drawn on top of
        each other, with pixel values equal to the road strength, from
        lowest to highest strength.

    """

    im = np.zeros(shape)

    for i in reversed(range(roads.shape[0])):
        strength, angle, distance, width = roads[i]
        coord = _get_line_box_cuts(angle, distance, *shape)
        if coord is None: continue # do not abort on bogus angle/distance
        coord = np.asarray(coord)
        x, y = _road_polygon(coord, width)
        rr, cc = polygon(y, x, shape)
        im[rr,cc] = strength

    return im


def find_roads(
        probability_map,
        *,
        input_threshold=0.3,
        max_roads=None,
        min_strength=0.17, #0.2,
        num_angles=720,
        roads_min_angle=np.pi/8,
        roads_min_distance=50,
        debugimage=None, # for debugging ...
        debugprint=None): # for debugging ...
    """
    Finds full-image roads in probability map (image).

    Parameters:
        probability_map -- an numpy.ndarray with probabilities per pixel (*)

    (*) i.e., the array is shaped HxW, with pixel values from 0 to 1

    Keyword-Only Parameters:
        input_threshold -- threshold applied to probability_map
        max_roads -- maximum number of roads to be found
        min_strength -- minimum strength of roads to be found
        num_angles -- angular resolution used in hough transforms
        roads_min_angle -- minimum required angle between roads
        roads_min_distance -- minimum required distance between roads

    Returns:
        roads -- roads that have been found (*)
        shape -- shape of probability_map (vector with 2 elements)

    (*) A numpy.ndarray with floating point type of shape Nx4, with N being
        the number of roads found, and 4 corresponding to columns 'strength',
        'angle', 'distance', 'width'. Strength is the response for the road
        (the "probability"), 'angle' and 'distance' correspond to the values
        returned by skimage.transform.hough_line, and 'width' is the
        identified road width (can currently be 12, 32 or 48).

    """

    # shorthand
    im = probability_map

    # the angles to be used in the Hough transform
    theta = np.linspace(-np.pi/2, np.pi/2, num_angles)

    # normalize almost anything to grayscale
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = im[:,:,:3] # throw away alpha
        im = im.mean(axis=2) # convert RGB to grayscale

    if debugimage: debugimage('original', im, 0, 1, 'jet')

    assert im.ndim == 2

    if debugimage:
        hspace, _, _ = hough_line(im, theta)
        debugimage('original_hough_hspace', hspace)

    # create monochrome/binary input map
    im[im >= input_threshold] = 1
    im[im < input_threshold] = 0

    if debugimage: debugimage('threshold_applied', im)

    # Hough transform
    hspace, angles, distances = hough_line(im, theta)

    hspace = np.asarray(hspace, dtype=np.float32)
    hspace /= hspace.max() # normalize

    if debugimage: debugimage('hough_hspace', hspace)

    # convolution filters, rectangular, tuned for widths of 12, 32, 48 pixels
    w12 = np.concatenate([-np.ones((6)), np.ones((12)), -np.ones((6))])
    w32 = np.concatenate([-np.ones((16)), np.ones((32)), -np.ones((16))])
    w48 = np.concatenate([-np.ones((24)), np.ones((48)), -np.ones((24))])

    # convolve
    im12 = ndi.filters.convolve1d(hspace, w12, axis=0)
    im32 = ndi.filters.convolve1d(hspace, w32, axis=0)
    im48 = ndi.filters.convolve1d(hspace, w48, axis=0)

    # normalize signal strengths for different road widths
    im12 /= 12
    im32 /= 32
    im48 /= 48

    ca = (None, None, 'jet',)
    if debugimage: debugimage('hough_hspace_conv12', im12, *ca)
    if debugimage: debugimage('hough_hspace_conv32', im32, *ca)
    if debugimage: debugimage('hough_hspace_conv48', im48, *ca)
    if debugimage:
        debugimage('hough_hspace_combined',
            np.hstack([im12, im32, im48]), *ca)

    # compute possible roads of all widths, sorted by signal strength
    seq = np.stack((im12, im32, im48)).flatten()
    sor = np.argsort(seq)
    roads = np.column_stack((
        seq,
        np.tile(np.tile(angles, distances.shape[0]), 3),
        np.tile(np.repeat(distances, angles.shape[0]), 3),
        np.repeat([12, 32, 48], distances.shape[0] * angles.shape[0])
    ))[sor][::-1]

    # columns: strength, angle, distance, width
    found_roads = np.asarray([]).reshape(0, 4)

    # find as many as strong roads as desired, while dropping roads that are too
    # similar to roads already found (non-max suppression)
    for i in range(roads.shape[0]):
        if roads[i,0] < min_strength:
            break
        a = roads[i,1]
        d = roads[i,2]
        close = (
            np.logical_or(
                np.logical_and(
                    np.abs(found_roads[:,1]-a) < roads_min_angle,
                    np.abs(found_roads[:,2]-d) < roads_min_distance),
                np.logical_and(
                    np.pi - np.abs(found_roads[:,1]-a) < roads_min_angle,
                    np.abs(found_roads[:,2]+d) < roads_min_distance)))
        if not np.any(close):
            found_roads = np.vstack((found_roads, roads[i]))
            if max_roads is not None and found_roads.shape[0] >= max_roads:
                break

    return found_roads, im.shape


# find begin and end coordinates of an intersection of a box (0, 0, width,
# height) with a line (given by angle and distance, as per Hough transform)
def _get_line_box_cuts(angle, distance, width, height):
    a = np.cos(angle)
    b = np.sin(angle)
    d = distance
    # TODO: handle divide-by-zero
    x0 = d/a
    x1 = (d-b*height)/a
    y0 = d/b
    y1 = (d-a*width)/b
    intersections = []
    if x0 >= 0 and x0 <= width: intersections.append((x0, 0))
    if x1 >= 0 and x1 <= width: intersections.append((x1, height))
    if y0 >= 0 and y0 <= height: intersections.append((0, y0))
    if y1 >= 0 and y1 <= height: intersections.append((width, y1))
    # TODO: what about degenerate cases?
    if len(intersections) == 0: return None
    assert len(intersections) == 2, (x0, x1, y0, y1)
    return intersections


# return a list of pixel coordinates, usable to index 2D ndarrays, that
# correspond to the shape of line segment with given width
def _road_polygon(endpoints, width):
    a, b = endpoints
    a = np.asarray(a)
    b = np.asarray(b)
    n = b-a
    n /= np.linalg.norm(n)
    n *= width / 2
    s = np.dot(np.array([[0, -1], [1, 0]]), n)
    xy = np.array([
        a - n - s,
        a - n + s,
        b + n + s,
        b + n - s
        ])
    x = xy[:,0]
    y = xy[:,1]
    return [x, y]
