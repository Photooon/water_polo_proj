import os
import cv2
import argparse
import numpy as np

from typing import Optional, List, Tuple
from detector import PlayerDetector
from homography import HomographyEstimator
from heatmap import generate_heatmap
from utils import get_imgs_from_dir_or_file

class WaterPoloTrackingPipeline:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.homography = HomographyEstimator(verbose=verbose)
        self.detector = PlayerDetector(verbose=verbose)
        self.prev_frame = None
    
    def process_frame(self, frame: np.ndarray, prev_frame: np.ndarray = None) -> dict:
        """
        Process a single frame with temporal tracking and spatial filtering.
        
        Args:
            frame: Current frame (BGR)
            prev_frame: Previous frame for optical flow tracking fallback
            
        Returns:
            Dictionary with detection results
        """
        # Detect pool with fallback to optical flow tracking
        success = self.homography.detect_with_fallback(frame, prev_frame)
        
        if not success:
            return {
                "success": False,
                "boxes": [],
                "confidences": [],
                "class_ids": [],
                "centroids": [],
                "transformed_positions": [],
                "using_tracked_corners": False
            }
        
        # Detect players
        boxes, confidences, class_ids = self.detector.detect(frame)
        
        # Filter detections to only those inside the pool
        pool_mask = self.homography.pool_mask
        boxes, confidences, class_ids = self.detector.filter_by_mask(
            boxes, confidences, class_ids, pool_mask
        )
        
        centroids = self.detector.get_centers(boxes)

        # Transform player positions to top-down pool view
        transformed_positions = []
        for center in centroids:
            transformed_point = self.homography.transform_point(center)
            if transformed_point:
                transformed_positions.append(transformed_point)
        
        # Detect ball (yellow circular object in pool)
        ball_center, ball_radius = self.detector.detect_ball(frame, pool_mask)
        ball_transformed = None
        if ball_center:
            ball_transformed = self.homography.transform_point(ball_center)
        
        result = {
            "success": True,
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "centroids": centroids,
            "transformed_positions": transformed_positions,
            "using_tracked_corners": self.homography.using_tracked_corners,
            "ball_center": ball_center,
            "ball_radius": ball_radius,
            "ball_transformed": ball_transformed
        }

        return result
    
    def process_image(self, image_path: str) -> dict:
        """Legacy method for backward compatibility."""
        frame = cv2.imread(image_path)
        result = self.process_frame(frame, self.prev_frame)
        self.prev_frame = frame.copy()
        return result

    def visualize_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                             class_ids: List[int], confidences: List[float], 
                             show_water: bool = False,
                             ball_center: Tuple[int, int] = None,
                             ball_radius: float = 0) -> np.ndarray:
        corner_frame = self.homography.visualize_corners(frame, show_water=show_water)
        
        # Add indicator if using tracked corners
        if self.homography.using_tracked_corners:
            cv2.putText(corner_frame, "TRACKED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        detect_frame = self.detector.visualize_detections(corner_frame, boxes, confidences, class_ids)
        
        # Draw ball if detected
        if ball_center:
            cv2.circle(detect_frame, ball_center, int(ball_radius), (0, 255, 255), 3)  # Yellow outline
            cv2.circle(detect_frame, ball_center, 5, (0, 255, 255), -1)  # Yellow center dot
            cv2.putText(detect_frame, "BALL", (ball_center[0] + 15, ball_center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return detect_frame

    def visualize_poolview(self, transformed_positions: List[Tuple[float, float]], 
                           ball_transformed: Tuple[float, float] = None) -> np.ndarray:
        pool_view = self.homography.get_pool_view()     # get blanket pool view

        pointed_view = self.homography.draw_on_pool_view(pool_view, transformed_positions, color=(255, 0, 0), radius=7)
        
        # Draw ball on pool view (yellow, larger)
        if ball_transformed:
            ball_pos = [ball_transformed]
            pointed_view = self.homography.draw_on_pool_view(pointed_view, ball_pos, color=(0, 255, 255), radius=10)
            # Add "BALL" label
            bx, by = int(ball_transformed[0]), int(ball_transformed[1])
            cv2.putText(pointed_view, "BALL", (bx + 12, by - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return pointed_view
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Polo Player Tracking Pipeline")
    parser.add_argument('input', type=str, help="Input image path or directory")
    parser.add_argument('--output', default="data/pipeline", type=str, help="Output directory")
    parser.add_argument('--show-water', action='store_true', 
                       help="Show detected water region as purple overlay")
    parser.add_argument('--show-extended', action='store_true',
                       help="Save extended canvas debug image")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Enable verbose logging output")
    parser.add_argument('--homography-data', default="data/homography_data", type=str, help="Directory to save homography JSONs")
    parser.add_argument('--detection-data', default="data/bounding_boxes", type=str, help="Directory to save detection JSONs")
    args = parser.parse_args()

    pipeline = WaterPoloTrackingPipeline(verbose=args.verbose)

    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Processing {len(image_files)} images...")

    os.makedirs(os.path.join(args.output, "detect"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "poolview"), exist_ok=True)

    prev_frame = None
    for idx, image_path in enumerate(image_files):
        if args.verbose:
            print(f"\n[{idx+1}/{len(image_files)}] Processing {image_path}")
        elif (idx + 1) % 5 == 0 or idx == len(image_files) - 1:
            print(f"  Processed {idx+1}/{len(image_files)} images...")
        
        frame = cv2.imread(image_path)
        result = pipeline.process_frame(frame, prev_frame)
        prev_frame = frame.copy()
        
        if not result["success"]:
            if args.verbose:
                print(f"\tSkipped - could not detect or track pool corners")
            continue

        vis_frame = pipeline.visualize_detections(frame, result["boxes"], result["class_ids"], 
                                                   result["confidences"], show_water=args.show_water,
                                                   ball_center=result.get("ball_center"),
                                                   ball_radius=result.get("ball_radius", 0))
        pool_view = pipeline.visualize_poolview(result["transformed_positions"], 
                                                ball_transformed=result.get("ball_transformed"))

        vis_image_path = os.path.join(args.output, "detect", os.path.basename(image_path))
        pool_view_path = os.path.join(args.output, "poolview", os.path.basename(image_path))
        cv2.imwrite(vis_image_path, vis_frame)
        cv2.imwrite(pool_view_path, pool_view)
        
        # Save extended canvas debug image if requested
        if args.show_extended and hasattr(pipeline.homography, 'debug_extended_mask') and pipeline.homography.debug_extended_mask is not None:
            os.makedirs(os.path.join(args.output, "extended"), exist_ok=True)
            ext_path = os.path.join(args.output, "extended", os.path.basename(image_path))
            cv2.imwrite(ext_path, pipeline.homography.debug_extended_mask)

        if args.verbose:
            status = "TRACKED" if result["using_tracked_corners"] else "DETECTED"
            num_players = sum(1 for c in result["class_ids"] if c == 0)
            print(f"\tCorners: {status}, Players: {num_players}")

        # Save data for heatmap
        if pipeline.homography.homography_matrix is not None:
             # Save homography data
            homography_data = {
                "image_path": image_path,
                "homography_matrix": pipeline.homography.homography_matrix.tolist(),
                "corners": pipeline.homography.corners.tolist() if pipeline.homography.corners is not None else [],
                "img_pool_points": pipeline.homography.img_pool_points.tolist() if pipeline.homography.img_pool_points is not None else []
            }
            
            h_json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
            h_json_path = os.path.join(args.homography_data, h_json_filename)
            with open(h_json_path, "w") as f:
                import json
                json.dump(homography_data, f, indent=4)

            # Save detection data
            detection_data = {
                "image_path": image_path,
                "boxes": [box.tolist() for box in result["boxes"]],
                "confidences": result["confidences"],
                "class_ids": result["class_ids"]
            }
            
            d_json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
            d_json_path = os.path.join(args.detection_data, d_json_filename)
            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(d_json_path), exist_ok=True)
            with open(d_json_path, "w") as f:
                json.dump(detection_data, f, indent=4)

        print(f"\tProcessed {image_path}")

    print("Pipeline processing completed.")
    
    # Generate heatmap
    heatmap_output = os.path.join(args.output, "player_heatmap.png")
    print(f"Generating heatmap to {heatmap_output}...")
    generate_heatmap(args.homography_data, args.detection_data, heatmap_output)
