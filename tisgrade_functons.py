
"""
sign_detection.py
-----------------
Utilities for detecting and geolocating traffic signs from panoramic street-level
imagery (e.g. Panoramax).
 
High-level pipeline
-------------------
1. For each annotated sign in a panoramic image, compute a *bearing line*: a
   ray cast from the camera position in the direction of the sign.
2. Intersect bearing lines from different viewpoints. Each intersection is a
   candidate sign location.
3. Cluster intersections with DBSCAN (recursive, with adaptive epsilon) to
   separate distinct signs from noise.
4. Score each cluster by spatial density,the ratio of observed intersections
   to the theoretical maximum and the amount of points in te cluster
   Then resolve line-ownership conflicts bases on this score so that
   each bearing line is assigned to at most one sign.
5. Compute a centroid, estimated physical sign size, and viewing direction for every
   accepted cluster.
6. Persist results to a PostGIS database and/or GeoPackage files.
 
Dependencies
------------
    numpy, pandas, geopandas, shapely, scipy, scikit-learn,
    geopy, haversine, psycopg2
"""
 
# Standard library
import math
# import os
# from collections import defaultdict
# import datetime 
# from pathlib import Path
 
# Third-party
import numpy as np
# from geopy.distance import geodesic
# from haversine import haversine
# from psycopg2.extras import execute_values
# from scipy.optimize import minimize          # imported but reserved for future use
from scipy.stats import circmean
# from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree
from sklearn.cluster import DBSCAN
# from sklearn.metrics.pairwise import haversine_distances
# import geopandas as gpd

import numpy as np
# from typing import List, Tuple

import shapely
from shapely import from_wkb, intersection, get_type_id, STRtree, GeometryType
# from typing import Dict
import pyproj

# local 
import tisgrade_classes as tsgc






# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
EARTH_RADIUS_M = 6_367_445  # Mean Earth radius in metres (used for unit conversions)
 

 
#  deel van deze functies mogelijk toevoegen aan de classe als static functions
 
 
# ===========================================================================
# Recursive clustering
# ===========================================================================

# def funct_cluster(
#     cluster: tsgc.Cluster,
#     max_radius_m: float = 3,
#     epsilon: float = 1.5,
#     # next_group_label: int = 1,
#     max_dept: int = 1,
#     current_dept: int = 1,
#     min_samples: int = 1,
# ) -> list[tsgc.Cluster]:
#     """Recursively cluster a set of line-intersection points into sign locations.
 
#     Cluster function:

#     1
#     Check in the list of intersections already match the criteria to form a cluster.

#     2 a
#     If this is the case create a defaultdict, set the dict value to next_group_label, add the items and return the list. end of fuction.

#     2 b
#     else call the cluster algorithm and create a defaultdict for each found cluster and add the intersections. First label is next_group_label, next cluser: next_group_label + 1 and so on.
#     The function applies DBSCAN with a haversine metric.

#     3
#     Remove all cluster that have are not valid (label -1)

#     4
#     Score the quality of all valid clusters

#     5
#     Loop to valid clusters
#         Loop to all intersections
#             Keep only those intersections that are not already in a other cluster

#     6
#     Remove cluster that don’t meet the minimum requirements any more

#     7
#     Decrease epsilon
#     After each level the
#     clusters are scored, conflicts between bearing lines are resolved (each
#     line may belong to at most one sign), and then *epsilon* is tightened by
#     25 % before recursing.


#     8
#     Create a dictionary list to add all the cluster to.
#     For each cluster
#         call this function recursive
#         Add the returned dictionary (label and intersections) to the dictionary list
#         Set next label to the highest label + 1

#     9
#     Return the dictionary clusters with their intersections
    

#     Recursion terminates when either:
 
#     * *current_dept* exceeds *max_dept*, **or**
#     * all points already lie within *max_radius_rad* of their centroid.
 
#     Parameters
#     ----------
#     cluster:
#         cluster. The cluster contains a List of object of the class Intersection
#     max_radius_rad:
#         Maximum allowed distance (in radians) from a cluster centroid.
#         Points closer than this threshold are already considered well-clustered.
#     epsilon:
#         DBSCAN neighbourhood radius in radians (≈ metres / EARTH_RADIUS_M).
#     next_group_label:
#         Integer label assigned to the first cluster at this recursion level.
#         Subsequent clusters get incrementing labels to avoid collisions across
#         recursive calls.
#     max_dept:
#         Maximum recursion depth.
#     current_dept:
#         Current recursion depth (internal counter; callers should leave at 1).
#     min_samples:
#         Minimum cluster size passed to DBSCAN and used to filter tiny clusters.
 
#     Returns
#     -------
#     List of clusters
#     """

#     # print(
#     #     f"\ncall to funct_cluster | n_points: {cluster.amount_intersections()}, "
#     #     f", epsilon: {epsilon}, depth: {current_dept}/{max_dept}"
#     # )

#     cluster_lst : list[tsgc.Cluster] = []
#     # --- Base case: maximum depth reached ---------------------------------- #
#     if current_dept > max_dept:
#         cluster_lst.append(cluster)
#         # result[next_group_label].extend(items)
#         print(f"  → max depth reached; returning cluster {cluster.get_id()} ({cluster.amount_intersections()} pts)")
#         return cluster_lst
 
#     current_dept += 1

 
#     # print(f"  max dist from centroid: {cluster.get_max_distance_intersection_center_m():.6f}  (threshold: {max_radius_m})")
 
#     if cluster.get_max_distance_intersection_center_m() < max_radius_m:
#         # All points are already within the acceptable radius → single cluster
#         cluster_lst.append(cluster)
#         # result[next_group_label].extend(items)
#         return cluster_lst
 
#     # --- DBSCAN clustering ------------------------------------------------- #
#     # Get coordinates and intersections (guaranteed same order and length)
#     coordinates_rad = cluster.get_coordinates_rad()
#     intersections = cluster.get_intersections()

#     clusterer = DBSCAN(
#         eps=epsilon*2*math.pi/EARTH_RADIUS_M,
#         min_samples=min_samples,
#         metric="haversine",
#         algorithm="ball_tree",
#     )
#     labels = clusterer.fit_predict(coordinates_rad)


#     # Create new clusters for each label (excluding noise label -1)
#     unique_labels = set(labels) - {-1}    
    
#     for label in unique_labels:
#         new_cluster = tsgc.Cluster()
#         # Add all intersections with current label
#         for idx, current_label in enumerate(labels):
#             if current_label == label:
#                 new_cluster.add_intersection(intersections[idx])
#         cluster_lst.append(new_cluster)

 
#     clusters_sorted = sorted(
#         cluster_lst,
#         key=lambda cluster: cluster.score_cluster(),
#         reverse=True
#     )


#     # --- Conflict resolution ----------------------------------------------- #
#     # Each bearing line may be assigned to at most one cluster.  We process
#     # clusters in descending score order so that high-quality clusters claim
#     # their lines first.
#     line_to_cluster: dict[int, int] = {}
#     # result_filterd : list[tsgc.Cluster] = []

#     # cluster_filtered: defaultdict = defaultdict(list)
 
#     for cluster in clusters_sorted:
#         cluster_id = cluster.get_id()
#         for inters in cluster.get_intersections():
#             id_a, id_b = inters.get_annotation1().get_id(), inters.get_annotation2().get_id()
 
#             # Skip if either line is already owned by a *different* cluster
#             conflict_a = id_a in line_to_cluster and line_to_cluster[id_a] != cluster_id
#             conflict_b = id_b in line_to_cluster and line_to_cluster[id_b] != cluster_id
#             if conflict_a or conflict_b:
#                 cluster.remove_intersection(inters)
 
#             line_to_cluster[id_a] = cluster_id
#             line_to_cluster[id_b] = cluster_id
 




# TILE_SIZE = 2000.0   # meters — tune based on profiling; bigger tiles = fewer tiles but more memory per tile
# BUFFER = 200.0        # meters, >= max line length: guarantees no missed cross-tile intersection


# def _local_aeqd_crs(lon_lat_coords: np.ndarray) -> pyproj.CRS:
#     """Build an azimuthal-equidistant CRS centered on the data's bounding box.

#     Antimeridian-safe: unwraps longitudes before finding the center so an
#     AOI straddling 180/-180 (e.g. Fiji, the Bering Strait) doesn't get a
#     bogus center from naive min/max.
#     """
#     lons, lats = lon_lat_coords[:, 0], lon_lat_coords[:, 1]

#     if lons.max() - lons.min() > 180:
#         lons = np.where(lons < 0, lons + 360, lons)

#     lon0 = (lons.min() + lons.max()) / 2.0
#     lon0 = ((lon0 + 180) % 360) - 180  # wrap back to [-180, 180]
#     lat0 = (lats.min() + lats.max()) / 2.0

#     return pyproj.CRS.from_proj4(
#         f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
#     )


# def _project_lines(lines: np.ndarray, transformer: pyproj.Transformer) -> np.ndarray:
#     """Bulk-reproject an array of shapely LineStrings via shapely.transform
#     (one vectorized pyproj call over all vertices, no per-geometry looping)."""

#     def _xy(coords: np.ndarray) -> np.ndarray:
#         x, y = transformer.transform(coords[:, 0], coords[:, 1])
#         return np.column_stack([x, y])

#     return shapely.transform(lines, _xy)


# def _crossing_pairs(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """Vectorized crossing-pair detection: STRtree for candidates, then
#     shapely ufuncs (no Python-level loop) to keep only true point crossings."""
#     tree = STRtree(lines)
#     i, j = tree.query(lines, predicate="crosses")

#     mask = j > i  # dedupe symmetric pairs
#     i, j = i[mask], j[mask]
#     if len(i) == 0:
#         return i, j

#     pts = shapely.intersection(lines[i], lines[j])
#     is_point = shapely.get_type_id(pts) == GeometryType.POINT
#     return i[is_point], j[is_point]


# def _process_tile(
#     tile_lines: np.ndarray,
#     tile_annotations: list,
#     core_minx: float,
#     core_miny: float,
#     core_maxx: float,
#     core_maxy: float,
# ) -> list[tuple]:
#     """Pure function, no closures over shared state — deliberately shaped so
#     it can be handed to ProcessPoolExecutor.map / Dask later without changes."""
#     i, j = _crossing_pairs(tile_lines)
#     if len(i) == 0:
#         return []

#     pts = shapely.intersection(tile_lines[i], tile_lines[j])
#     xs, ys = shapely.get_x(pts), shapely.get_y(pts)

#     # Only keep intersections whose point falls in this tile's unbuffered
#     # core — this is what prevents double-counting the same intersection
#     # from two overlapping buffered tiles.
#     in_core = (
#         (xs >= core_minx) & (xs < core_maxx) &
#         (ys >= core_miny) & (ys < core_maxy)
#     )
#     i, j = i[in_core], j[in_core]

#     return [(tile_annotations[a], tile_annotations[b]) for a, b in zip(i, j)]


# def find_intersections(annotations: list["tsgc.ObjectAnnotation"]) -> list["tsgc.Intersection"]:
#     """Find all pairwise crossing points between bearing lines from ObjectAnnotations.

#     Lines are given in lon/lat (EPSG:4326). They're reprojected into a local,
#     dynamically-centered azimuthal-equidistant CRS so tile/buffer math can be
#     done in meters, then processed tile-by-tile to bound peak memory.

#     Valid for AOIs roughly up to 100km x 100km, latitude within [-60, 80]
#     (outside that range AEQD distortion grows and this approach should be
#     revisited). Currently runs single-process; _process_tile is written as a
#     pure function so it can be swapped into a process pool or Dask later.
#     """
#     if not annotations:
#         return []

#     valid_lines = []
#     valid_annotations = []
#     for annotation in annotations:
#         line = annotation.get_line()
#         if line is not None and line.is_valid and line.geom_type == "LineString":
#             valid_lines.append(line)
#             valid_annotations.append(annotation)

#     if not valid_lines:
#         return []

#     lines_arr = np.array(valid_lines, dtype=object)

#     # --- reproject lon/lat -> local meters ---
#     all_coords = shapely.get_coordinates(lines_arr)  # (N, 2) lon, lat
#     local_crs = _local_aeqd_crs(all_coords)
#     transformer = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
#     projected_lines = _project_lines(lines_arr, transformer)

#     # --- tile the projected extent ---
#     bounds = shapely.bounds(projected_lines)  # (N, 4): minx, miny, maxx, maxy
#     minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
#     maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

#     intersections = []
#     x = minx
#     while x < maxx:
#         y = miny
#         while y < maxy:
#             core_minx, core_miny = x, y
#             core_maxx, core_maxy = x + TILE_SIZE, y + TILE_SIZE

#             sel = (
#                 (bounds[:, 2] >= core_minx - BUFFER) & (bounds[:, 0] <= core_maxx + BUFFER) &
#                 (bounds[:, 3] >= core_miny - BUFFER) & (bounds[:, 1] <= core_maxy + BUFFER)
#             )
#             idx = np.where(sel)[0]

#             if len(idx) > 1:
#                 pairs = _process_tile(
#                     projected_lines[idx],
#                     [valid_annotations[k] for k in idx],
#                     core_minx, core_miny, core_maxx, core_maxy,
#                 )
#                 for a, b in pairs:
#                     intersections.append(tsgc.Intersection(a, b))

#             y += TILE_SIZE
#         x += TILE_SIZE

#     return intersections