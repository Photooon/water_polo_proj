import os
import cv2
import config
import argparse
import numpy as np

from typing import List, Tuple, Optional
from utils import get_imgs_from_dir_or_file

class HomographyEstimator:
    def __init__(self):
        self.homography_matrix = None
        self.img_pool_points = None
        
        # Define pool corners in top-down view (in meters)
        # Order: top-left, top-right, bottom-right, bottom-left
        # Manually define left and right pool halves
        self.left_pool_points = np.float32([
            [0, 0],
            [config.POOL_LENGTH // 2, 0],
            [config.POOL_LENGTH // 2, config.POOL_WIDTH],
            [0, config.POOL_WIDTH]
        ])

        self.right_pool_points = np.float32([
            [config.POOL_LENGTH // 2, 0],
            [config.POOL_LENGTH, 0],
            [config.POOL_LENGTH, config.POOL_WIDTH],
            [config.POOL_LENGTH // 2, config.POOL_WIDTH]
        ])
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pool boundaries and compute homography matrix
        Using color segmentation to find water area and extract corners

        Args:
            frame: Input image as numpy array (BGR)
        Returns:
            Homography matrix or None if detection fails
        """       
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_water = np.array([80, 50, 50])
        upper_water = np.array([140, 255, 255])
        water_color_range = (lower_water, upper_water)
        
        mask = cv2.inRange(hsv, water_color_range[0], water_color_range[1])
        
        # Connect all water area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            raise ValueError("No water area detected in the frame")
        
        largest_contour = max(contours, key=cv2.contourArea)
        # area = cv2.contourArea(largest_contour)

        pool_mask = np.zeros_like(mask)
        cv2.drawContours(pool_mask, [largest_contour], -1, 255, -1)

        # Get corners from the largest water area
        perimeter = cv2.arcLength(largest_contour, True)
        
        epsilon = 0.02 * perimeter
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        corners = approx_polygon.reshape(-1, 2)
        num_corners = len(corners)
        
        if num_corners < 3:
            raise ValueError("No enough corners detected for pool boundary.")
        
        # Solve boundries
        boundary_lines = []
        
        for i in range(num_corners):
            p1 = corners[i]
            p2 = corners[(i + 1) % num_corners]
            
            line = (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            length = np.linalg.norm(p2 - p1)
            
            direction = (p2 - p1) / length if length > 0 else np.array([0, 0])
            
            boundary_lines.append({
                'coords': line,
                'length': length,
                'direction': direction,
                'p1': i,
                'p2': (i + 1) % num_corners,
                'index': i,
            })
        
        # Manually define top corner as the one with smallest y value, simplifying assumption
        top_corner_idx = np.argmin(corners[:, 1])
        
        edge1 = boundary_lines[(top_corner_idx - 1) % num_corners]
        edge2 = boundary_lines[top_corner_idx]

        edge1_p2 = edge1["p2"] if edge1["p1"] == top_corner_idx else edge1["p1"]
        edge2_p2 = edge2["p2"] if edge2["p1"] == top_corner_idx else edge2["p1"]

        vertical_edge = edge1 if corners[edge1_p2][1] > corners[edge2_p2][1] else edge2
        horizontal_edge = edge2 if vertical_edge == edge1 else edge1

        top_corner_idx2 = horizontal_edge["p1"] if horizontal_edge["p1"] != top_corner_idx else horizontal_edge["p2"]
        bottom_corner_idx = vertical_edge["p1"] if vertical_edge["p1"] != top_corner_idx else vertical_edge["p2"]

        # find another bottom corner by traversing along the contour
        direction = 1 if bottom_corner_idx - top_corner_idx == 1 or (bottom_corner_idx == 0 and top_corner_idx == num_corners - 1) else -1
        next_idx = (bottom_corner_idx + direction + num_corners) % num_corners
        while next_idx != top_corner_idx:
            corner = corners[next_idx]
            if frame.shape[0] - corner[1] < config.BOTTOM_MARGIN:
                bottom_corner_idx2 = next_idx
                break

            next_idx = (next_idx + direction + num_corners) % num_corners

        top_left_idx = top_corner_idx if corners[top_corner_idx][0] < corners[top_corner_idx2][0] else top_corner_idx2
        top_right_idx = top_corner_idx2 if top_left_idx == top_corner_idx else top_corner_idx
        bottom_right_idx = bottom_corner_idx if corners[bottom_corner_idx][0] > corners[bottom_corner_idx2][0] else bottom_corner_idx2
        bottom_left_idx = bottom_corner_idx2 if bottom_right_idx == bottom_corner_idx else bottom_corner_idx

        self.img_pool_points = np.float32([
            corners[top_left_idx],
            corners[top_right_idx],
            corners[bottom_right_idx],
            corners[bottom_left_idx]
        ])

        self.homography_matrix, _ = cv2.findHomography(
            self.img_pool_points, 
            self.left_pool_points if top_left_idx == top_corner_idx else self.right_pool_points,
            method=cv2.RANSAC
        )

    def transform_point(self, point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Transform a single pixel coordinate to pool coordinate
        
        Args:
            point: (x, y) pixel coordinate
            
        Returns:
            (x, y) pool coordinate in meters, or None if no homography
        """
        # Convert point to homogeneous coordinates
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        return tuple(transformed[0][0])
    
    def visualize_corners(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw detected corners on frame
        
        Args:
            frame: Input image
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw corners in empty yellow circles and connect corners with lines
        for i, corner in enumerate(self.img_pool_points):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(annotated_frame, (x, y), 8, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"C{i}", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if i > 0:
                prev_corner = self.img_pool_points[i - 1]
                cv2.line(annotated_frame, (int(prev_corner[0]), int(prev_corner[1])), 
                         (x, y), (0, 255, 255), 2)
                
        # Connect last corner to first
        first_corner = self.img_pool_points[0]
        last_corner = self.img_pool_points[-1]
        cv2.line(annotated_frame, (int(last_corner[0]), int(last_corner[1])), 
                 (int(first_corner[0]), int(first_corner[1])), (0, 255, 255), 2)
        
        return annotated_frame

    def get_pool_view(self, width: int = 1000, height: int = 667) -> np.ndarray:
        """
        Create a blank top-down pool view canvas
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            
        Returns:
            Blank pool view image
        """
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw pool boundary
        cv2.rectangle(canvas, (0, 0), (width-1, height-1), (0, 0, 0), 2)
        
        # Draw center line
        cv2.line(canvas, (width//2, 0), (width//2, height-1), (200, 200, 200), 1)
        
        # Draw goal areas (2m lines)
        goal_line_x = int(2.0 / config.POOL_LENGTH * width)
        cv2.line(canvas, (goal_line_x, 0), (goal_line_x, height-1), (200, 200, 200), 1)
        cv2.line(canvas, (width - goal_line_x, 0), (width - goal_line_x, height-1), 
                (200, 200, 200), 1)
        
        return canvas
    
    def draw_on_pool_view(self, canvas: np.ndarray, pool_points: List[Tuple[float, float]], 
                         color: Tuple[int, int, int] = (255, 0, 0), radius: int = 5):
        """
        Draw points on the top-down pool view
        
        Args:
            canvas: Pool view image
            pool_points: List of (x, y) pool coordinates in meters
            color: BGR color tuple
            radius: Point radius
            
        Returns:
            Canvas with points drawn
        """
        height, width = canvas.shape[:2]
        
        for point in pool_points:
            px = int(point[0] / config.POOL_LENGTH * width)
            py = int(point[1] / config.POOL_WIDTH * height)
            
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(canvas, (px, py), radius, color, -1)
        
        return canvas
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detech pool boundaries and estimate homography')
    parser.add_argument('input', type=str, help='Input image path or directory')
    parser.add_argument('--output', default="data/homography", type=str, help='Output image dir')
    
    args = parser.parse_args()

    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Found {len(image_files)} images to process.")

    estimator = HomographyEstimator()

    os.makedirs(args.output, exist_ok=True)

    for image_path in image_files:
        frame = cv2.imread(image_path)

        estimator.detect(frame)
        annotated_frame = estimator.visualize_corners(frame)
        
        output_path = os.path.join(args.output, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_frame)
        print(f"\tProcessed {image_path}")

    print("Homography estimation completed.")