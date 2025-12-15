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
    def __init__(self):
        self.homography = HomographyEstimator()
        self.detector = PlayerDetector()
    
    def process_image(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        
        # Detect pool and construct homography
        self.homography.detect(frame)

        # Detect players
        boxes, confidences, class_ids = self.detector.detect(frame)
        centroids = self.detector.get_centers(boxes)

        # Transform player positions to top-down pool view
        transformed_positions = []
        for center in centroids:
            transformed_point = self.homography.transform_point(center)
            if transformed_point:
                transformed_positions.append(transformed_point)
        
        result = {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "centroids": centroids,
            "transformed_positions": transformed_positions
        }

        return result

    def visualize_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                             class_ids: List[int], confidences: List[float]) -> np.ndarray:
        corner_frame = self.homography.visualize_corners(frame)
        detect_frame = self.detector.visualize_detections(corner_frame, boxes, confidences, class_ids)
        return detect_frame

    def visualize_poolview(self, transformed_positions: List[Tuple[float, float]]) -> np.ndarray:
        pool_view = self.homography.get_pool_view()     # get blanket pool view

        pointed_view = self.homography.draw_on_pool_view(pool_view, transformed_positions, color=(255, 0, 0), radius=7)

        return pointed_view
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Polo Player Tracking Pipeline")
    parser.add_argument('input', type=str, help="Input image path or directory")
    parser.add_argument('--output', default="data/pipeline", type=str, help="Output directory")
    parser.add_argument('--homography-data', default="data/homography_data", type=str, help="Directory to save homography JSONs")
    parser.add_argument('--detection-data', default="data/bounding_boxes", type=str, help="Directory to save detection JSONs")
    args = parser.parse_args()

    pipeline = WaterPoloTrackingPipeline()

    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Found {len(image_files)} images to process.")

    os.makedirs(os.path.join(args.output, "detect"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "poolview"), exist_ok=True)
    os.makedirs(args.homography_data, exist_ok=True)
    os.makedirs(args.detection_data, exist_ok=True)

    for image_path in image_files:
        result = pipeline.process_image(image_path)

        frame = cv2.imread(image_path)
        vis_frame = pipeline.visualize_detections(frame, result["boxes"], result["class_ids"], result["confidences"])
        pool_view = pipeline.visualize_poolview(result["transformed_positions"])

        vis_image_path = os.path.join(args.output, "detect", os.path.basename(image_path))
        pool_view_path = os.path.join(args.output, "poolview", os.path.basename(image_path))
        cv2.imwrite(vis_image_path, vis_frame)
        cv2.imwrite(pool_view_path, pool_view)

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
            with open(d_json_path, "w") as f:
                json.dump(detection_data, f, indent=4)

        print(f"\tProcessed {image_path}")

    print("Pipeline processing completed.")
    
    # Generate heatmap
    heatmap_output = os.path.join(args.output, "player_heatmap.png")
    print(f"Generating heatmap to {heatmap_output}...")
    generate_heatmap(args.homography_data, args.detection_data, heatmap_output)