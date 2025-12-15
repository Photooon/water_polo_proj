import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import argparse
from typing import List, Tuple

def load_json_data(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def get_common_frames(homography_dir: str, detections_dir: str) -> List[str]:
    homography_files = set(f for f in os.listdir(homography_dir) if f.endswith('.json'))
    detection_files = set(f for f in os.listdir(detections_dir) if f.endswith('.json'))
    
    common_files = sorted(list(homography_files.intersection(detection_files)))
    return common_files

def transform_points(points: List[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    if not points:
        return np.array([])
        
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)

def generate_heatmap(homography_dir: str, detections_dir: str, output_path: str):
    common_frames = get_common_frames(homography_dir, detections_dir)
    print(f"Found {len(common_frames)} frames with both homography and detection data.")
    
    all_pool_points = []
    
    for filename in common_frames:
        homography_path = os.path.join(homography_dir, filename)
        detection_path = os.path.join(detections_dir, filename)
        
        h_data = load_json_data(homography_path)
        d_data = load_json_data(detection_path)
        
        # Get homography matrix
        H = np.array(h_data['homography_matrix'])
        
        # Get player centers
        centers = []
        # Filter for players (class_id 0)
        for box, class_id in zip(d_data['boxes'], d_data['class_ids']):
            if class_id == 0:  # Player
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
        
        if not centers:
            continue
            
        # Transform points
        pool_points = transform_points(centers, H)
        
        # Filter points within pool bounds
        for x, y in pool_points:
            if 0 <= x <= config.POOL_LENGTH and 0 <= y <= config.POOL_WIDTH:
                all_pool_points.append((x, y))
                
    if not all_pool_points:
        print("No valid player positions found.")
        return

    points_array = np.array(all_pool_points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Draw pool boundary
    plt.plot([0, config.POOL_LENGTH, config.POOL_LENGTH, 0, 0], 
             [0, 0, config.POOL_WIDTH, config.POOL_WIDTH, 0], 'k-', linewidth=2)
    
    # Draw center line
    plt.plot([config.POOL_LENGTH/2, config.POOL_LENGTH/2], [0, config.POOL_WIDTH], 'k--', alpha=0.5)
    
    # Draw 2m lines
    plt.plot([2, 2], [0, config.POOL_WIDTH], 'r:', alpha=0.5)
    plt.plot([config.POOL_LENGTH-2, config.POOL_LENGTH-2], [0, config.POOL_WIDTH], 'r:', alpha=0.5)
    
    # Plot heatmap
    # We invert Y axis because usually image coordinates are top-down, but plots are bottom-up.
    # However, our pool coordinates are defined with (0,0) at top-left or similar.
    # Let's check homography.py: 
    # self.left_pool_points = [[0, 0], [25, 0], [25, 21], [0, 21]]
    # So (0,0) is one corner, (50, 21) is opposite.
    # Standard plot has (0,0) at bottom-left. 
    # To match the "image" feel where (0,0) is top-left, we might want to invert Y.
    # But for a map, standard Cartesian is fine. Let's stick to standard Cartesian but label appropriately.
    # Actually, let's invert Y to match the "top-down view" convention if we want it to look like the pool diagram.
    
    sns.kdeplot(x=x_coords, y=y_coords, fill=True, cmap="viridis", alpha=0.7, levels=20, thresh=0.05)
    
    plt.scatter(x_coords, y_coords, c='red', s=5, alpha=0.3, label='Individual Detections')
    
    plt.xlim(-2, config.POOL_LENGTH + 2)
    plt.ylim(config.POOL_WIDTH + 2, -2) # Invert Y axis to match image coordinate system feel
    
    plt.title(f"Player Position Heatmap (n={len(all_pool_points)})")
    plt.xlabel("Length (m)")
    plt.ylabel("Width (m)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate player heatmap from detection and homography data")
    parser.add_argument("--homography-dir", default="data/homography_data", help="Directory containing homography JSONs")
    parser.add_argument("--detections-dir", default="data/bounding_boxes", help="Directory containing detection JSONs")
    parser.add_argument("--output", default="data/player_heatmap.png", help="Output path for heatmap image")
    
    args = parser.parse_args()
    
    generate_heatmap(args.homography_dir, args.detections_dir, args.output)
