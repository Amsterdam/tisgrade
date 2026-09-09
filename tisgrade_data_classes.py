
from dataclasses import dataclass

@dataclass
class GeoBounds:
    """Geographic bounding box for the area of interest"""
    longitude_min: float  # x min
    longitude_max: float  # x max
    latitude_min: float   # y min
    latitude_max: float   # y max

@dataclass
class SignParameters:
    """Settings for the sign type and size"""
    sign_regex: str         # regex for the sign exampels '^NL:C21' '^NL:A01-30.*', '^NL:[ABCD].*', '^NL:(?![A-I]).*'
    sign_size_min: float    # expected minimum sign size
    sign_size_max: float    # expected maximum sign size
    sign_size_margin: float # margin on the min and max sign size. 0 is no margin, 1 is 100% margin


@dataclass
class RunParameters:
    """Parameters of the run"""    
    run_start_date_time: str    # start periode to look for the objects
    run_end_date_time: str      # end periode to look for the objects
    run_name: str               # name of the run eg: 'Test run', 'Run Amsterdam 2026

@dataclass
class QualitySettings:
    """setting for the qualty of the intersections and the clusters"""
    intersection_angle_deg_min: float   # Below this the location estimate is unstable
    intersection_angle_deg_max: float   # Above this one camera likely couldn't see the sign
    sign_size_score: float              # Relative size discrepancy between the two lines
    cluster_radius_m_max: float         # Maximum radius (metres) within which points are considered part of the same sign location.
    cluster_min_size: int               # Minimum size of a cluster. min value 1, measn one intersction is enhough to create a cluster
    cluster_max_dept: int               # Maximum dept of recursion of cluster algoritm
    cluster_epsilon_start: float        # Starting value of epsilon
    cluster_epsilon_decrease: float              # The factor with epsilon gets smaller each time a recursion occurs
