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
POOL_LENGTH = 50.0  # Standard water polo pool length
POOL_WIDTH = 21.0   # Standard water polo pool width

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
