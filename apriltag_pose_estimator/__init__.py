"""
AprilTag-based 2D global pose estimation for tour guide robots.

A modular, production-ready Python package for estimating 2D robot pose
(X, Y, Yaw) using AprilTags with known global coordinates.

Key components:
- PoseEstimator: Main pose estimation engine
- CameraCalibration: Camera intrinsics and distortion handling
- AprilTagMap: Map of known tag positions
- PoseEstimate: 2D pose output with confidence
- pose fusion: Multi-tag detection fusion
- Utilities: Angle handling, rotation conversions

Usage:
    from apriltag_pose_estimator import PoseEstimator, CameraCalibration, AprilTagMap
    
    calib = CameraCalibration.from_yaml("config/camera_calibration.yaml")
    tags = AprilTagMap.from_json("config/tag_map.json")
    estimator = PoseEstimator(calib, tags)
    
    pose = estimator.estimate_pose(frame)
    print(f"Robot at ({pose.x:.2f}, {pose.y:.2f}), yaw={pose.yaw:.2f}, conf={pose.confidence:.2f}")
"""

__version__ = "1.0.0"
__author__ = "AprilTag Tour Guide Team"

from .camera_calibration import CameraCalibration
from .fusion import PoseEstimate, fuse_poses, fuse_poses_with_median
from .pose_estimator import PoseEstimator, TagDetection
from .tag_map import AprilTagMap, TagDefinition
from . import utils

__all__ = [
    "PoseEstimator",
    "TagDetection",
    "CameraCalibration",
    "AprilTagMap",
    "TagDefinition",
    "PoseEstimate",
    "fuse_poses",
    "fuse_poses_with_median",
    "utils",
]
