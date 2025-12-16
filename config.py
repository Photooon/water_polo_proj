"""
Configuration file for Water Polo Tracking System
"""

# Homography parameters
BOTTOM_MARGIN = 50  # Pixels from the lowest bottom corner to the img bottom

# Detection parameters
DETECTION_CONFIDENCE = 0.2  # YOLO confidence threshold
DETECTION_IOU = 0.5  # Non-maximum suppression IOU threshold
YOLO_MODEL = "yolov10x.pt"  # Use larger model for better accuracy

# Tracking parameters
MAX_DISAPPEARED = 30  # Max frames before removing a track
MAX_DISTANCE = 100  # Max pixel distance for matching detections

# Pool dimensions (in meters)
# 25 yards = 22.86m (length), 42 feet = 12.8m (width)
POOL_LENGTH = 22.86  # 25 yards
POOL_WIDTH = 12.8    # 42 feet

# Camera setup
CAMERA_ANGLE_DEG = 45.0  # Camera angle relative to pool edge in degrees
CAMERA_HEIGHT = 3.0      # Camera height in meters

# Pool detection tuning
# Fraction of pool length visible in frame (0.5 = half the pool visible)
# The visible near edge is the full pool width (42 feet)
# The visible pool extends about this fraction of the pool length
VISIBLE_POOL_FRACTION = 0.5

# Visualization parameters
HEATMAP_BINS = 50
TRAJECTORY_ALPHA = 0.6
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

# Video processing
FRAME_SKIP = 1  # Process every N frames (1 = process all frames)
VIDEO_OUTPUT_FPS = 30
