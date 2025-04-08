"""
========
zones.py
========

:summary:
    Handles the representation and method connected with zones/areas.
    This module contains a family of classes used for representing
    and querying zones. This has been designed for reuse, with various
    child area-polygon classes and their associated group handlers.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

from typing import List
from abc import ABC

import numpy as np


def _assert_lat_lon_valid(lat_points: np.ndarray, lon_points: np.ndarray):
    """
    Checks that coordinates of latitude and longitude points are valid.
    This function compares the size of the two and checks that both
    are equal one dimensional arrays.

    :param lat_points: 1D latitude points given in decimal degrees.
    :type lat_points: np.ndarray

    :param lon_points: 1D longitude points given in decimal degrees.
    :type lon_points: np.ndarray
    """

    if lat_points.ndim != 1:
        raise ValueError("lat_points must be a 1D array")

    if lon_points.ndim != 1:
        raise ValueError("lon_points must be a 1D array")

    if np.shape(lat_points) != np.shape(lon_points):
        raise ValueError("lat_points and lon_points must have same shape")


class Polygon(ABC):
    """
    Represents a simple latitude-longitude closed polygon.
    Can be used for defining custom 2D zones on the globe.
    """

    def __init__(self,
                 lat_points: np.ndarray,
                 lon_points: np.ndarray,
                 priority: int = 0):
        """
        Defines a polygon given latitude and longitude points and priority.
        If coordinates for an open polygon are given it will be automatically
        closed using the first and last points.

        :param lat_points: A list of latitude coordinates in decimal degrees.
            This must be one dimensional and the same length as lon_points.
        :type lat_points: np.ndarray

        :param lon_points: A list of longitude coordinates in decimal degrees.
            This must be one dimensional and the same length as lat_points.
        :type lat_points: np.ndarray

        :param priority: The priority/layer order of the polygon. Polygons
            with higher priority will be used over overlapping lower
            priority polygons.
        :type priority: int
        """

        _assert_lat_lon_valid(lat_points, lon_points)
        self._lat_points = lat_points
        self._lon_points = lon_points
        self._priority = priority

        # If the given polygon is not closed:
        if ((self._lat_points[0] != self._lat_points[-1]) or
                (self._lon_points[0] != self._lon_points[-1])):

            # Automatically add point to close it.
            self._lat_points = np.append(self._lat_points, self._lat_points[0])
            self._lon_points = np.append(self._lon_points, self._lon_points[0])

    def is_point_inside(self, lat_point: float, lon_point: float) -> bool:
        """
        Determines if a given position is inside the polygon.
        Given the latitude and longitude coordinates of a single point in
        decimal degrees, this function will return True if the point is
        inside the polygon.

        :param lat_point: The latitude coordinate of the point of interest.
        :type lat_point: float

        :param lon_point: The longitude coordinate of the point of interest.
        :type lon_point: float

        :return: Returns True if given point is inside the polygon.
        :rtype: bool
        """
        return is_inside_polygon(
            lon_point, lat_point, self._lon_points, self._lat_points)

    def is_points_inside(self, lat_points: np.ndarray, lon_points: np.ndarray) -> np.ndarray:
        """
        Determines if a series of positions are inside the polygon.
        Given the latitude and longitude coordinates of a multiple points
        in decimal degrees, this function will return True for each point
        that is inside the polygon.

        :param lat_points: The latitude coordinates of the points of interest.
        :type lat_points: float

        :param lon_points: The longitude coordinates of the points of interest.
        :type lon_points: float

        :return: Returns True for each given point inside the polygon.
        :rtype: bool
        """
        _assert_lat_lon_valid(lat_points, lon_points)
        return is_inside_polygon_vec(
            lon_points, lat_points, self._lon_points, self._lat_points)

    @property
    def lat_points(self) -> np.ndarray:
        """
        Returns the list of latitude coordinate points of the polygon.
        This property returns a copy, preventing accidental editing.

        :return: The latitude coordinate points of the polygon.
        :rtype: np.ndarray
        """
        return np.copy(self._lat_points)

    @property
    def lon_points(self) -> np.ndarray:
        """
        Returns the list of longitude coordinate points of the polygon.
        This property returns a copy, preventing accidental editing.

        :return: The longitude coordinate points of the polygon.
        :rtype: np.ndarray
        """
        return np.copy(self._lon_points)

    @property
    def priority(self) -> int:
        """
        Returns the priority (or layer) of the polygon.
        This is used for determining the outcome when multiple polygons
        overlapping one another are involved. When this occurs the highest
        priority polygon is used.

        :return: The priority (or layer) of the polygon.
        :rtype: int
        """
        return self._priority

class FlaggedPolygon(Polygon):
    """
    Represents a latitude-longitude closed polygon with binary active flag.
    These polygons are used for determining "active" and "inactive" areas.
    For example, areas where specific sensors, such as where GPS or
    quantum sensors can and cannot be used.
    """

    def __init__(self, lat_points: np.ndarray, lon_points: np.ndarray,
                 is_active_area: bool, priority: int = 0):
        """
        Defines a polygon given latitude and longitude points, active flag
        and optional priority. If coordinates for an open polygon are given
        it will be automatically closed using the first and last points.

        :param lat_points: A list of latitude coordinates in decimal degrees.
            This must be one dimensional and the same length as lon_points.
        :type lat_points: np.ndarray

        :param lon_points: A list of longitude coordinates in decimal degrees.
            This must be one dimensional and the same length as lat_points.
        :type lat_points: np.ndarray

        :param is_active_area: If the polygon should represent an active
            area or not. For example, representing a GPS active zone.
        :type is_active_area: bool

        :param priority: The priority/layer order of the polygon. Polygons
            with higher priority will be used over overlapping lower
            priority polygons.
        :type priority: int
        """
        super().__init__(lat_points, lon_points, priority)
        self._is_active_area = is_active_area

    @property
    def is_active_area(self) -> bool:
        """
        Returns True if the polygon is active area.
        This can be used to determine if points which are found inside the
        polygon are to be considered inside an active area.

        :return: Whether the polygon is an active area.
        :rtype: bool
        """
        return self._is_active_area


class FlaggedPolygonGroup:
    """
    A handler for a group of flagged polygons.
    Allows the querying of multiple polygons, in priority order,
    with a default flag.
    """

    def __init__(self, polygons: List[FlaggedPolygon],
                 is_active_by_default: bool = True):
        """
        Creates group of flagged polygons with default is active behaviour.
        Handles list of given polygons, applying operations in priority order.
        If given positions do not fall inside any polygons, the default
        flag will be returned (True by default).

        :param polygons: A list of flagged polygons.
        :type polygons: List[FlaggedPolygon]

        :param is_active_by_default: The default flag to be returned.
        :type is_active_by_default: bool
        """
        self._polygons = sorted(polygons, key=lambda p: p.priority)
        self._is_active_by_default = is_active_by_default

    def get_point_flag(self, lat_point: float, lon_point: float) -> bool:
        """
        Returns the boolean flag for given point.
        Searches through given polygons in priority order and returns the
        flag for the highest priority polygon that the point is inside of.
        If the point is not in any polygons, the default flag will be used.

        :param lat_point: The latitude coordinate of the point of interest.
        :type lat_point: float

        :param lon_point: The longitude coordinate of the point of interest.
        :type lon_point: float

        :return: Boolean flag of the highest priority polygon the point is
            inside of (or the default flag for the group).
        :rtype: bool
        """

        # Set the return flag as the default flag
        to_return = self._is_active_by_default

        # Search through polygon's in order:
        for polygon in self._polygons:

            # If inside current polygon, update the flag
            if polygon.is_point_inside(lat_point, lon_point):
                to_return = polygon.is_active_area

        # Return the final flag
        return to_return

    @property
    def polygons(self) -> List[FlaggedPolygon]:
        """
        Returns list of flagged polygons used by group.

        :return: list of flagged polygons used by group
        :rtype: List[FlaggedPolygon]
        """
        return self._polygons

    @property
    def num_polygons(self) -> int:
        """
        Returns the number of polygons covered by the group.

        :return: The number of polygons covered by the group.
        :rtype: int
        """
        return len(self._polygons)

    def get_point_flags(self, lat_points: np.ndarray, lon_points: np.ndarray) -> np.ndarray:
        """
        Returns the boolean flags for multiple given points.
        Searches through given polygons in priority order and returns the
        flags of the highest priority polygons that each of the point is
        inside of. For points that do not fall inside nay polygon's the
        default flag is used.

        :param lat_points: The latitude coordinates of the points of interest.
        :type lat_points: np.ndarray

        :param lon_points: The longitude coordinates of the points of interest.
        :type lon_points: np.ndarray

        :return: Boolean flags of the highest priority polygon each point is
            inside of (or the default flag for the group).
        :rtype: np.ndarray
        """

        # Get the size of the inputs
        num_lat_points = lat_points.size
        num_lon_points = lon_points.size

        # Validate the input arrays (same size and dimensions)
        if num_lat_points != num_lon_points or (lat_points.size != lon_points.size):
            raise ValueError('lat_points and lon_points must have same 1-dimensional size')

        # Set the return flag as the default flag
        if self._is_active_by_default:
            to_return = np.ones(num_lat_points, dtype=bool)
        else:
            to_return = np.zeros(num_lat_points, dtype=bool)

        # Search through polygon's in order:
        for polygon in self._polygons:

            # If inside current polygon, update the flag
            tmp = polygon.is_points_inside(lat_points, lon_points)
            if np.any(tmp):
                to_return[tmp] = polygon.is_active_area

        # Return the final flag
        return to_return


def is_inside_polygon(x_point: float,
                      y_point: float,
                      polygon_x: np.ndarray,
                      polygon_y: np.ndarray) -> bool:
    """
    Determines if a given position is inside given the polygon.
    Tests by generating horizontal lines and assessing the number of
    polygon lines that are crossed. Using this count, a point can be
    determine to be inside the closed polygon or not.

    :param x_point: The x coordinate of the point of interest.
    :type x_point: float

    :param y_point: The y coordinate of the point of interest.
    :type y_point: float

    :param polygon_x: The x-coordinates of the polygon. Assumed to be closed.
    :type polygon_x: np.ndarray

    :param polygon_y: The y-coordinates of the polygon. Assumed to be closed.
    :type polygon_y: np.ndarray

    :return: Returns True if the given position is inside the polygon.
    :rtype: bool
    """

    inside = False
    num_vertices = len(polygon_x)

    # Get the first point on the polygon
    p1_x = float(polygon_x[0])
    p1_y = float(polygon_y[0])

    # Loop through each edge in the polygon
    for i in range(1, num_vertices + 1):

        # Get the next point in the polygon
        p2_x = float(polygon_x[i % num_vertices])
        p2_y = float(polygon_y[i % num_vertices])

        # Check if the point is above the minimum y coordinate of the edge
        if y_point > min(p1_y, p2_y):

            # Check if the point is below the maximum y coordinate of the edge
            if y_point <= max(p1_y, p2_y):

                # Check if the point is to the left of the maximum x coordinate of the edge
                if x_point <= max(p1_x, p2_x):

                    # Calculate the x-intersection of the line connecting the point to the edge
                    x_intersection = (y_point - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x

                    # Check if the point is on the same line as the edge or to the left of the x-intersection
                    if p1_x == p2_x or x_point <= x_intersection:

                        # Flip the inside flag
                        inside = not inside

        # Store the current point as the first point for the next iteration
        p1_x = p2_x
        p1_y = p2_y

    # Return the value of the inside flag
    return inside


def is_inside_polygon_vec(x_points: np.ndarray,
                          y_points: np.ndarray,
                          polygon_x: np.ndarray,
                          polygon_y: np.ndarray) -> np.ndarray:
    """
    Determines if a given position is inside given the polygon.
    Tests by generating horizontal lines and assessing the number of
    polygon lines that are crossed. Using this count, points can be
    determine to be inside the closed polygon or not.

    :param x_points: The x coordinates of the point of interest.
    :type x_points: float

    :param y_points: The y coordinates of the point of interest.
    :type y_points: float

    :param polygon_x: The x-coordinates of the polygon. Assumed to be closed.
    :type polygon_x: np.ndarray

    :param polygon_y: The y-coordinates of the polygon. Assumed to be closed.
    :type polygon_y: np.ndarray

    :return: Returns True if the given position is inside the polygon.
    :rtype: bool
    """

    inside = np.zeros(x_points.size, dtype=bool)
    num_vertices = len(polygon_x)

    p1_x = float(polygon_x[0])
    p1_y = float(polygon_y[0])

    # Loop through each edge in the polygon
    for i in range(1, num_vertices + 1):

        # Get the next point in the polygon
        p2_x = float(polygon_x[i % num_vertices])
        p2_y = float(polygon_y[i % num_vertices])

        # For points that are between the line's Y coordinates
        # and left of the line's maximum x coordinate:
        check = ((y_points > min(p1_y, p2_y))
                 & (y_points <= max(p1_y, p2_y))
                 & (x_points <= max(p1_x, p2_x)))

        if np.any(check):

            # Calculate the x-intersection of the line connecting the point to the edge
            x_intersection = (y_points[check] - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x

            # Check if the point is on the same line as the edge or to the left of the x-intersection
            tmp = x_points[check] <= x_intersection
            tmp |= p1_x == p2_x

            # Flip the corresponding flags
            inside[check] ^= tmp

        # Store the current point as the first point for the next iteration
        p1_x = p2_x
        p1_y = p2_y

    # Return the value of the inside flag
    return inside

