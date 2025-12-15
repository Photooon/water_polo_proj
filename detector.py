import os
import cv2
import json
import config
import argparse
import numpy as np

from ultralytics import YOLO
from typing import List, Tuple
from utils import get_imgs_from_dir_or_file


class PlayerDetector:
    def __init__(self, 
                 model_path: str = config.YOLO_MODEL, 
                 conf: float = config.DETECTION_CONFIDENCE,
                 iou: float = config.DETECTION_IOU):
        self.model = YOLO(model_path)
        self.confidence_threshold = conf
        self.iou_threshold = iou
        
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input image as numpy array (BGR)
            
        Returns:
            Tuple of (bounding_boxes, confidences, class_ids)
            - bounding_boxes: List of [x1, y1, x2, y2] coordinates
            - confidences: List of confidence scores
            - class_ids: List of detected class IDs
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Extract detections
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Filter for person class (class 0 in COCO)
                # We can also include sports ball (class 32) for ball detection
                if cls == 0 or cls == 32:
                    boxes.append(np.array([x1, y1, x2, y2]))
                    confidences.append(conf)
                    class_ids.append(cls)
        
        return boxes, confidences, class_ids
    
    def get_centers(self, boxes: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Calculate center points of bounding boxes
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            List of (cx, cy) center coordinates
        """
        centers = []
        for box in boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            centers.append((cx, cy))
        return centers
    
    def visualize_detections(self, frame: np.ndarray, boxes: List[np.ndarray], 
                             confidences: List[float], class_ids: List[int]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input image
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Different colors for players and ball
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            label = f"Player {conf:.2f}" if cls == 0 else f"Ball {conf:.2f}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Detect players and ball in water polo images')
    parser.add_argument('input', type=str, help='Input image path or directory')
    parser.add_argument('--output', default="data/detections", type=str, help='Output directory')
    parser.add_argument('--data-output', default="data/bounding_boxes", type=str, help='Output directory for detection data')
    parser.add_argument('--conf', type=float, default=config.DETECTION_CONFIDENCE, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Found {len(image_files)} images to process.")
    
    detector = PlayerDetector(conf=args.conf)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.data_output, exist_ok=True)
    
    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(image_path)
        
        boxes, confidences, class_ids = detector.detect(frame)

        # Save detection data
        detection_data = {
            "image_path": image_path,
            "boxes": [box.tolist() for box in boxes],
            "confidences": confidences,
            "class_ids": class_ids
        }
        
        json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
        json_path = os.path.join(args.data_output, json_filename)
        with open(json_path, "w") as f:
            json.dump(detection_data, f, indent=4)
        
        annotated_frame = detector.visualize_detections(frame, boxes, confidences, class_ids)
        
        num_players = sum(1 for cls in class_ids if cls == 0)
        num_balls = sum(1 for cls in class_ids if cls == 32)
        print(f"\tDetected {num_players} players and {num_balls} balls in {image_path}")
        
        output_path = os.path.join(args.output, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_frame)
    
    print(f"Detection completed")