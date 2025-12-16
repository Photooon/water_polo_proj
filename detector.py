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
                 iou: float = config.DETECTION_IOU,
                 verbose: bool = False):
        self.model = YOLO(model_path)
        self.confidence_threshold = conf
        self.iou_threshold = iou
        self.verbose = verbose
        
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
    
    def detect_ball(self, frame: np.ndarray, mask: np.ndarray = None) -> Tuple[Tuple[int, int], float]:
        """
        Detect yellow water polo ball using HSV color filtering
        
        Args:
            frame: Input image as numpy array (BGR)
            mask: Optional pool mask to limit search area
            
        Returns:
            Tuple of (center, radius) if ball found, else (None, 0)
            - center: (cx, cy) coordinates
            - radius: estimated ball radius
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow color range in HSV
        # Yellow has Hue around 20-40 in OpenCV (0-180 scale)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([45, 255, 255])
        
        # Create yellow mask
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply pool mask if provided
        if mask is not None:
            yellow_mask = cv2.bitwise_and(yellow_mask, mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0
        
        # Find the best ball candidate based on:
        # 1. Circularity (how round it is)
        # 2. Size (reasonable ball size)
        best_ball = None
        best_score = 0
        
        # Minimum confidence threshold - must be very confident it's a ball
        MIN_CONFIDENCE = 0.65
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Skip very small or very large contours
            # Ball should be roughly 200-3000 pixels depending on distance
            if area < 100 or area > 5000:
                continue
            
            # Calculate circularity (1.0 = perfect circle)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Require high circularity (very round objects only)
            if circularity < 0.65:
                continue
            
            # Score based on circularity and size
            # Weight circularity heavily since ball should be very round
            size_score = min(area / 500, 1.0)  # Normalize size
            score = circularity * 0.8 + size_score * 0.2
            
            if score > best_score and score >= MIN_CONFIDENCE:
                best_score = score
                # Get bounding circle
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                best_ball = (int(cx), int(cy)), radius
        
        if best_ball:
            return best_ball
        return None, 0
    
    def filter_by_mask(self, boxes: List[np.ndarray], confidences: List[float], 
                       class_ids: List[int], mask: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Filter detections to keep only those inside a mask (e.g., pool water region).
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            class_ids: List of class IDs
            mask: Binary mask where 255 = inside, 0 = outside
            
        Returns:
            Filtered (boxes, confidences, class_ids) tuples
        """
        if mask is None:
            return boxes, confidences, class_ids
        
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        
        centers = self.get_centers(boxes)
        
        for box, conf, cls, center in zip(boxes, confidences, class_ids, centers):
            cx, cy = center
            
            # Check if center point is inside the mask
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                if mask[cy, cx] > 0:
                    filtered_boxes.append(box)
                    filtered_confidences.append(conf)
                    filtered_class_ids.append(cls)
        
        num_filtered = len(boxes) - len(filtered_boxes)
        if num_filtered > 0 and self.verbose:
            print(f"Filtered out {num_filtered} detections outside pool area")
        
        return filtered_boxes, filtered_confidences, filtered_class_ids
    
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