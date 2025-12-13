import os
import cv2
import argparse
import numpy as np

from typing import Optional, List, Tuple
from detector import PlayerDetector
from homography import HomographyEstimator
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
    args = parser.parse_args()

    pipeline = WaterPoloTrackingPipeline()

    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Found {len(image_files)} images to process.")

    os.makedirs(os.path.join(args.output, "detect"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "poolview"), exist_ok=True)

    for image_path in image_files:
        result = pipeline.process_image(image_path)

        frame = cv2.imread(image_path)
        vis_frame = pipeline.visualize_detections(frame, result["boxes"], result["class_ids"], result["confidences"])
        pool_view = pipeline.visualize_poolview(result["transformed_positions"])

        vis_image_path = os.path.join(args.output, "detect", os.path.basename(image_path))
        pool_view_path = os.path.join(args.output, "poolview", os.path.basename(image_path))
        cv2.imwrite(vis_image_path, vis_frame)
        cv2.imwrite(pool_view_path, pool_view)

        print(f"\tProcessed {image_path}")

    print("Pipeline processing completed.")