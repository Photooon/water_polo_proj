import os
import cv2
import json
import config
import argparse
import numpy as np

from typing import List, Tuple, Optional
from utils import get_imgs_from_dir_or_file

class HomographyEstimator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.homography_matrix = None
        self.img_pool_points = None
        self.pool_mask = None  # Binary mask of water region
        
        # Temporal tracking state
        self.prev_frame = None
        self.prev_corners = None
        self.using_tracked_corners = False  # True if corners were tracked, not detected
        self.pool_side = "full"  # "left", "right", or "full"
        
        # Define pool corners in top-down view (in meters)
        # Order: top-left, top-right, bottom-right, bottom-left
        
        # Full pool view (for smaller pools where camera sees entire pool)
        self.full_pool_points = np.float32([
            [0, 0],
            [config.POOL_LENGTH, 0],
            [config.POOL_LENGTH, config.POOL_WIDTH],
            [0, config.POOL_WIDTH]
        ])
        
        # Left half pool (for larger pools with camera on one side)
        self.left_pool_points = np.float32([
            [0, 0],
            [config.POOL_LENGTH / 2, 0],
            [config.POOL_LENGTH / 2, config.POOL_WIDTH],
            [0, config.POOL_WIDTH]
        ])

        # Right half pool
        self.right_pool_points = np.float32([
            [config.POOL_LENGTH / 2, 0],
            [config.POOL_LENGTH, 0],
            [config.POOL_LENGTH, config.POOL_WIDTH],
            [config.POOL_LENGTH / 2, config.POOL_WIDTH]
        ])
    
    def get_pool_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get binary mask of the water/pool region for filtering detections.
        
        Args:
            frame: Input image as numpy array (BGR)
            
        Returns:
            Binary mask where 255 = inside pool, 0 = outside
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_water = np.array([80, 50, 50])
        upper_water = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest contour (main pool area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        self.pool_mask = mask
        return mask
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pool boundaries using color transition edges.
        Finds the dominant water color, extends pool edges to find corners,
        and optimizes to minimize blue pixels outside the detected quad.

        Args:
            frame: Input image as numpy array (BGR)
        Returns:
            Homography matrix or None if detection fails
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find the dominant color in the image (should be water blue)
        # Sample the center region to avoid edges
        center_hsv = hsv[h//4:3*h//4, w//4:3*w//4]
        
        # Look for blue-ish hues (around 80-140 in HSV)
        blue_mask_sample = cv2.inRange(center_hsv, 
                                        np.array([80, 30, 30]), 
                                        np.array([140, 255, 255]))
        
        # Get the median hue of blue pixels for accurate color detection
        blue_pixels = center_hsv[blue_mask_sample > 0]
        if len(blue_pixels) > 0:
            median_hue = np.median(blue_pixels[:, 0])
            median_sat = np.median(blue_pixels[:, 1])
            # Create dynamic range around detected water color
            hue_range = 20
            lower_water = np.array([max(0, median_hue - hue_range), 30, 30])
            upper_water = np.array([min(180, median_hue + hue_range), 255, 255])
        else:
            # Fallback to default blue range
            lower_water = np.array([80, 30, 30])
            upper_water = np.array([140, 255, 255])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        total_blue_pixels = np.sum(water_mask > 0)
        if self.verbose:
            print(f"Total blue pixels: {total_blue_pixels}")
        
        # Helper function for line intersection
        def line_intersection(line1, line2):
            """Find intersection of two parametric lines (vx, vy, x0, y0)"""
            vx1, vy1, x1, y1 = line1
            vx2, vy2, x2, y2 = line2
            denom = vx1 * vy2 - vy1 * vx2
            if abs(denom) < 1e-10:
                return None
            t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
            return np.array([x1 + t * vx1, y1 + t * vy1])
        
        def is_in_frame(pt, h, w):
            return 0 <= pt[0] <= w and 0 <= pt[1] <= h
        
        def is_on_water(pt, mask):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                return mask[y, x] > 0
            return False
        
        def ray_extend(p1, p2, length_scale=5.0):
            """Extend ray from p1 through p2 by length_scale times the distance"""
            direction = p2 - p1
            return p2 + direction * length_scale
        
        # Get water contour
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No water contour found")
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get polygon approximation of the contour
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.01 * perimeter  # Tight approximation to get all corners
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        poly_corners = approx.reshape(-1, 2).astype(np.float32)
        
        if self.verbose:
            print(f"Polygon has {len(poly_corners)} corners")
            print("All polygon corners (R#):")
            for i, pt in enumerate(poly_corners):
                print(f"  R{i}: [{pt[0]:.0f}, {pt[1]:.0f}]")
        
        # Compute coverage function with noise metric
        def compute_quad_metrics(corners_4, mask):
            quad = np.array(corners_4, dtype=np.int32)
            quad_mask = np.zeros_like(mask)
            cv2.fillPoly(quad_mask, [quad], 255)
            
            blue_inside = np.sum((mask > 0) & (quad_mask > 0))   # Water inside quad
            blue_outside = np.sum((mask > 0) & (quad_mask == 0)) # Water outside quad (missed)
            noise = np.sum((mask == 0) & (quad_mask > 0))        # Non-water inside quad (noise)
            
            return blue_inside, blue_outside, noise
        
        # Strategy: Use line intersection to find pool corners
        # R0, R7 are on the left/top edge; R4, R5, R6 are on bottom/right edges
        
        # Get key polygon points
        r0 = poly_corners[np.argmin(poly_corners[:, 0])]  # leftmost
        r7 = poly_corners[np.argmin(poly_corners[:, 1])]  # topmost
        
        # Find edge points
        edge_margin = 20
        bottom_mask = poly_corners[:, 1] >= h - edge_margin
        right_mask = poly_corners[:, 0] >= w - edge_margin
        
        bottom_pts = poly_corners[bottom_mask] if np.any(bottom_mask) else poly_corners
        right_pts = poly_corners[right_mask] if np.any(right_mask) else poly_corners
        
        r4 = bottom_pts[np.argmin(bottom_pts[:, 0])]  # leftmost on bottom
        r5 = right_pts[np.argmax(right_pts[:, 1])]    # bottommost on right (BR)
        r6 = right_pts[np.argmin(right_pts[:, 1])]    # topmost on right
        
        if self.verbose:
            print(f"Key points: R0=[{r0[0]:.0f},{r0[1]:.0f}] R7=[{r7[0]:.0f},{r7[1]:.0f}] R4=[{r4[0]:.0f},{r4[1]:.0f}] R5=[{r5[0]:.0f},{r5[1]:.0f}] R6=[{r6[0]:.0f},{r6[1]:.0f}]")
        
        # Line intersection helper
        def line_intersection_2d(p1, p2, p3, p4):
            """Find intersection of line through (p1,p2) and line through (p3,p4)"""
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            x3, y3 = float(p3[0]), float(p3[1])
            x4, y4 = float(p4[0]), float(p4[1])
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                return None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)
            return [ix, iy]
        
        # Ray extension helper
        def extend_ray_to_boundary(start, through, ext_h, ext_w, pad_x, pad_y):
            """Extend ray from start through 'through' to extended canvas boundary"""
            # Convert to extended canvas coords
            s = np.array([start[0] + pad_x, start[1] + pad_y], dtype=float)
            t = np.array([through[0] + pad_x, through[1] + pad_y], dtype=float)
            
            direction = t - s
            if np.linalg.norm(direction) < 1e-6:
                return [through[0], through[1]]
            direction = direction / np.linalg.norm(direction)
            
            # Find intersection with extended canvas boundaries
            t_values = []
            
            if direction[0] > 1e-6:  # Right boundary
                t_val = (ext_w - 1 - s[0]) / direction[0]
                if t_val > 0:
                    y = s[1] + t_val * direction[1]
                    if 0 <= y <= ext_h - 1:
                        t_values.append((t_val, [ext_w-1 - pad_x, y - pad_y]))
            
            if direction[0] < -1e-6:  # Left boundary
                t_val = -s[0] / direction[0]
                if t_val > 0:
                    y = s[1] + t_val * direction[1]
                    if 0 <= y <= ext_h - 1:
                        t_values.append((t_val, [-pad_x, y - pad_y]))
            
            if direction[1] > 1e-6:  # Bottom boundary
                t_val = (ext_h - 1 - s[1]) / direction[1]
                if t_val > 0:
                    x = s[0] + t_val * direction[0]
                    if 0 <= x <= ext_w - 1:
                        t_values.append((t_val, [x - pad_x, ext_h-1 - pad_y]))
            
            if direction[1] < -1e-6:  # Top boundary
                t_val = -s[1] / direction[1]
                if t_val > 0:
                    x = s[0] + t_val * direction[0]
                    if 0 <= x <= ext_w - 1:
                        t_values.append((t_val, [x - pad_x, -pad_y]))
            
            if t_values:
                t_values.sort(key=lambda x: x[0])
                return t_values[0][1]
            return [through[0], through[1]]
        
        import config
        pool_ratio = config.POOL_LENGTH / config.POOL_WIDTH  # 22.86/12.8 = 1.79
        visible_fraction = getattr(config, 'VISIBLE_POOL_FRACTION', 0.5)
        
        candidates = []
        
        # Helper to order corners as TL, TR, BR, BL
        def order_corners(corners):
            arr = np.array(corners)
            sums = arr[:, 0] + arr[:, 1]
            diffs = arr[:, 0] - arr[:, 1]
            tl = corners[np.argmin(sums)]
            tr = corners[np.argmax(diffs)]
            br = corners[np.argmax(sums)]
            bl = corners[np.argmin(diffs)]
            return [list(tl), list(tr), list(br), list(bl)]
        
        # Get key points from polygon corners
        r0 = poly_corners[np.argmin(poly_corners[:, 0])]  # leftmost (BL visible)
        r7 = poly_corners[np.argmin(poly_corners[:, 1])]  # topmost (TL visible)
        
        # Find edge points for side directions
        edge_margin = 20
        bottom_mask = poly_corners[:, 1] >= h - edge_margin
        right_mask = poly_corners[:, 0] >= w - edge_margin
        bottom_pts = poly_corners[bottom_mask] if np.any(bottom_mask) else poly_corners
        right_pts = poly_corners[right_mask] if np.any(right_mask) else poly_corners
        r4 = bottom_pts[np.argmin(bottom_pts[:, 0])]  # leftmost on bottom edge
        r6 = right_pts[np.argmin(right_pts[:, 1])]    # topmost on right edge
        
        if self.verbose:
            print(f"Key points: R0=[{r0[0]:.0f},{r0[1]:.0f}] R7=[{r7[0]:.0f},{r7[1]:.0f}] R4=[{r4[0]:.0f},{r4[1]:.0f}] R6=[{r6[0]:.0f},{r6[1]:.0f}]")
        
        # Calculate near edge length and side directions
        near_edge_len = np.linalg.norm(np.array([r7[0] - r0[0], r7[1] - r0[1]]))
        
        dir_r0_r4 = np.array([r4[0] - r0[0], r4[1] - r0[1]], dtype=float)
        dir_r7_r6 = np.array([r6[0] - r7[0], r6[1] - r7[1]], dtype=float)
        len_r0_r4 = np.linalg.norm(dir_r0_r4)
        len_r7_r6 = np.linalg.norm(dir_r7_r6)
        
        dir_r0_r4_unit = dir_r0_r4 / len_r0_r4 if len_r0_r4 > 0 else np.array([1, 0])
        dir_r7_r6_unit = dir_r7_r6 / len_r7_r6 if len_r7_r6 > 0 else np.array([1, 0])
        
        # Visible side length (average of the two visible sides)
        visible_side_len = (len_r0_r4 + len_r7_r6) / 2
        
        # Target side length: if visible fraction is 0.5, we see 50% of the pool
        # So the full pool side = visible_side / visible_fraction
        target_side = visible_side_len / visible_fraction if visible_fraction > 0 else visible_side_len
        
        if self.verbose:
            print(f"Pool geometry:")
            print(f"  Near edge (R0-R7): {near_edge_len:.0f}px = {config.POOL_WIDTH:.1f}m")
            print(f"  Visible side: {visible_side_len:.0f}px (~{visible_fraction:.0%} of pool)")
            print(f"  Target side: {target_side:.0f}px (full pool length)")
            print(f"  Pool ratio check: {target_side/near_edge_len:.2f} (expected {pool_ratio:.2f})")
        
        # Candidate 1: Simple visible corners (R0, R7, rightmost, bottommost)
        r_right = poly_corners[np.argmax(poly_corners[:, 0])]
        r_bottom = poly_corners[np.argmax(poly_corners[:, 1])]
        simple_corners = [
            [float(r7[0]), float(r7[1])],       # TL
            [float(r_right[0]), float(r_right[1])],  # TR (rightmost)
            [float(r_bottom[0]), float(r_bottom[1])], # BR (bottommost)
            [float(r0[0]), float(r0[1])]        # BL
        ]
        candidates.append(("visible_corners", order_corners(simple_corners)))
        
        # Candidate 2: Ratio-extended from R0/R7 along side directions
        tl = [float(r7[0]), float(r7[1])]
        tr = [r7[0] + dir_r7_r6_unit[0] * target_side, r7[1] + dir_r7_r6_unit[1] * target_side]
        br = [r0[0] + dir_r0_r4_unit[0] * target_side, r0[1] + dir_r0_r4_unit[1] * target_side]
        bl = [float(r0[0]), float(r0[1])]
        candidates.append(("ratio_extended", [tl, tr, br, bl]))
        
        # ========== EVALUATE CANDIDATES ==========
        best_corners = None
        best_score = float('inf')
        best_name = ""
        best_coverage = 0
        best_missed = 0
        best_noise = 0
        
        if self.verbose:
            print(f"\nEvaluating {len(candidates)} quad candidates:")
        for name, corners in candidates:
            inside, missed, noise = compute_quad_metrics(corners, water_mask)
            # Score: minimize missed + noise (maximize blue, minimize white)
            score = missed + noise
            coverage_pct = inside / total_blue_pixels * 100 if total_blue_pixels > 0 else 0
            if self.verbose:
                print(f"  {name}: coverage={coverage_pct:.1f}%, missed={missed}, noise={noise}, score={score}")
            
            if score < best_score:
                best_corners = corners
                best_score = score
                best_name = name
                best_coverage = inside
                best_missed = missed
                best_noise = noise
        
        if self.verbose:
            print(f"Selected: {best_name} (score={best_score})")
        
            # Print selected corners
            print(f"\nSelected corners ({best_name}):")
            labels = ['TL', 'TR', 'BR', 'BL']
            for label, pt in zip(labels, best_corners):
                in_frame = 0 <= pt[0] < w and 0 <= pt[1] < h
                status = "IN-FRAME" if in_frame else "VIRTUAL"
                if in_frame:
                    on_water = is_on_water(np.array(pt), water_mask)
                    status += f", on_water={on_water}"
                print(f"  {label}: [{pt[0]:.1f}, {pt[1]:.1f}] ({status})")
        
        # Create debug visualization
        pad_x, pad_y = w, h  # 3x canvas
        debug_ext = np.zeros((3*h, 3*w, 3), dtype=np.uint8)  # Black background
        debug_ext[pad_y:pad_y+h, pad_x:pad_x+w] = [255, 255, 255]  # White for in-frame
        # Color water blue
        for y in range(h):
            for x in range(w):
                if water_mask[y, x] > 0:
                    debug_ext[y+pad_y, x+pad_x] = [255, 200, 100]
        # Draw frame boundary
        cv2.rectangle(debug_ext, (pad_x, pad_y), (pad_x+w, pad_y+h), (0, 255, 0), 3)
        # Draw key points R0, R4, R5, R6, R7
        key_pts = {'R0': r0, 'R4': r4, 'R5': r5, 'R6': r6, 'R7': r7}
        for name, pt in key_pts.items():
            cv2.circle(debug_ext, (int(pt[0]+pad_x), int(pt[1]+pad_y)), 10, (0, 165, 255), -1)
        # Draw the quad
        quad_ext = [[pt[0]+pad_x, pt[1]+pad_y] for pt in best_corners]
        box_int = np.int32(quad_ext)
        cv2.drawContours(debug_ext, [box_int], 0, (0, 0, 255), 3)
        for i, pt in enumerate(quad_ext):
            cv2.circle(debug_ext, (int(pt[0]), int(pt[1])), 12, (255, 0, 0), -1)
        self.debug_extended_mask = debug_ext
        
        if best_corners is None:
            raise ValueError("Could not find valid corner combination")
        
        corners = best_corners
        coverage_pct = (best_coverage / total_blue_pixels * 100) if total_blue_pixels > 0 else 0
        
        # Report corners
        if self.verbose:
            corner_names = ['C0', 'C1', 'C2', 'C3']
            for name, pt in zip(corner_names, corners):
                pt_arr = np.array(pt)
                in_frame = is_in_frame(pt_arr, h, w)
                status = "IN-FRAME" if in_frame else "VIRTUAL"
                if in_frame:
                    on_water = is_on_water(pt_arr, water_mask)
                    status += f", on_water={on_water}"
                print(f"  {name}: {pt} ({status})")
        
            print(f"Coverage: {coverage_pct:.1f}% | Missed: {best_missed} | Noise: {best_noise}")
        
        self.img_pool_points = np.float32(corners)
        
        # Store visible contour corners for visualization
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        self.corners = approx.reshape(-1, 2)
        self.water_mask = water_mask  # Store for visualization
        self.detected_color = (lower_water, upper_water)  # Store detected color range
        
        if self.verbose:
            print(f"Visible corners: {len(self.corners)}")
        
        # Compute homography
        self.pool_side = "full"
        self.homography_matrix, _ = cv2.findHomography(
            self.img_pool_points, 
            self.full_pool_points,
            method=cv2.RANSAC
        )
        
        self.using_tracked_corners = False
        self.prev_corners = self.img_pool_points.copy()

    def _track_corners_opticalflow(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                                    prev_corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Track pool corners from previous frame to current frame using optical flow.
        
        Args:
            prev_frame: Previous frame (BGR)
            curr_frame: Current frame (BGR)
            prev_corners: Pool corners from previous frame (4x2 array)
            
        Returns:
            Tracked corners in current frame, or None if tracking fails
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare points for optical flow (needs shape Nx1x2)
        prev_pts = prev_corners.reshape(-1, 1, 2).astype(np.float32)
        
        # Lucas-Kanade optical flow parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )
        
        # Check if all points were tracked successfully
        if curr_pts is None or status is None:
            return None
        
        if not all(status.flatten()):
            if self.verbose:
                print("Warning: Some corner points could not be tracked")
            return None
        
        tracked_corners = curr_pts.reshape(-1, 2)
        return tracked_corners
    
    def detect_with_fallback(self, frame: np.ndarray, prev_frame: np.ndarray = None) -> bool:
        """
        Detect pool corners with fallback to optical flow tracking.
        
        First attempts auto-detection. If that fails and a previous frame
        with known corners is available, uses optical flow to track the corners.
        
        Args:
            frame: Current frame (BGR)
            prev_frame: Previous frame for tracking fallback (optional)
            
        Returns:
            True if corners were found (detected or tracked), False otherwise
        """
        # Generate pool mask for this frame (used for detection filtering)
        self.get_pool_mask(frame)
        
        # Try auto-detection first
        try:
            self.detect(frame)
            self.prev_frame = frame.copy()
            return True
        except ValueError as e:
            if self.verbose:
                print(f"Auto-detection failed: {e}")
        
        # Fallback to tracking if we have previous data
        if prev_frame is not None and self.prev_corners is not None:
            if self.verbose:
                print("Attempting optical flow tracking from previous frame...")
            tracked_corners = self._track_corners_opticalflow(
                prev_frame, frame, self.prev_corners
            )
            
            if tracked_corners is not None:
                self.img_pool_points = tracked_corners.astype(np.float32)
                
                # Recompute homography with tracked corners
                target_points = self.left_pool_points if self.pool_side == "left" else self.right_pool_points
                self.homography_matrix, _ = cv2.findHomography(
                    self.img_pool_points, target_points, method=cv2.RANSAC
                )
                
                self.using_tracked_corners = True
                self.prev_corners = self.img_pool_points.copy()
                self.prev_frame = frame.copy()
                self.corners = self.img_pool_points  # For visualization
                
                if self.verbose:
                    print("Successfully tracked corners using optical flow")
                return True
        
        # Check if we have stored previous frame from earlier detection
        elif self.prev_frame is not None and self.prev_corners is not None:
            if self.verbose:
                print("Attempting optical flow tracking from stored previous frame...")
            tracked_corners = self._track_corners_opticalflow(
                self.prev_frame, frame, self.prev_corners
            )
            
            if tracked_corners is not None:
                self.img_pool_points = tracked_corners.astype(np.float32)
                
                target_points = self.left_pool_points if self.pool_side == "left" else self.right_pool_points
                self.homography_matrix, _ = cv2.findHomography(
                    self.img_pool_points, target_points, method=cv2.RANSAC
                )
                
                self.using_tracked_corners = True
                self.prev_corners = self.img_pool_points.copy()
                self.prev_frame = frame.copy()
                self.corners = self.img_pool_points
                
                if self.verbose:
                    print("Successfully tracked corners using optical flow")
                return True
        
        if self.verbose:
            print("Warning: Could not detect or track pool corners")
        return False

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
    
    def visualize_corners(self, frame: np.ndarray, show_water: bool = False) -> np.ndarray:
        """
        Draw detected corners on frame with optional water region overlay
        
        Args:
            frame: Input image
            show_water: If True, highlight detected water region in purple
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Show detected water region as semi-transparent overlay
        if show_water and hasattr(self, 'water_mask') and self.water_mask is not None:
            # Create a purple overlay for the water
            water_overlay = np.zeros_like(annotated_frame)
            water_overlay[:] = (255, 0, 255)  # Purple/Magenta in BGR
            
            # Apply overlay only where water is detected
            mask_3ch = cv2.cvtColor(self.water_mask, cv2.COLOR_GRAY2BGR) > 0
            annotated_frame = np.where(mask_3ch, 
                                        cv2.addWeighted(annotated_frame, 0.7, water_overlay, 0.3, 0),
                                        annotated_frame)
            
            # Add text showing detected color info
            if hasattr(self, 'detected_color'):
                lower, upper = self.detected_color
                color_text = f"Water Hue: {int(lower[0])}-{int(upper[0])}"
                cv2.putText(annotated_frame, color_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw detected contour corners WITH LABELS
        h_frame, w_frame = frame.shape[:2]
        for idx, (x, y) in enumerate(self.corners):
            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Adjust label position based on corner location
            label_x = int(x) + 8
            label_y = int(y) + 5
            
            # If near right edge, put label to the left
            if x > w_frame - 50:
                label_x = int(x) - 35
            # If near left edge, put label to the right (already default)
            
            # If near top edge, put label below
            if y < 30:
                label_y = int(y) + 20
            # If near bottom edge, put label above
            if y > h_frame - 30:
                label_y = int(y) - 10
            
            cv2.putText(annotated_frame, f"R{idx}", (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
    parser.add_argument('--data-output', default="data/homography_data", type=str, help='Output directory for homography data')
    
    args = parser.parse_args()

    image_files = get_imgs_from_dir_or_file(args.input)
    print(f"Found {len(image_files)} images to process.")

    estimator = HomographyEstimator()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.data_output, exist_ok=True)

    for image_path in image_files:
        frame = cv2.imread(image_path)

        try:
            estimator.detect(frame)
        except Exception as e:
            print(f"\tSkipping {image_path}: {e}")
            continue
        
        # Save homography data
        if estimator.homography_matrix is not None:
            homography_data = {
                "image_path": image_path,
                "homography_matrix": estimator.homography_matrix.tolist(),
                "corners": estimator.corners.tolist() if estimator.corners is not None else [],
                "img_pool_points": estimator.img_pool_points.tolist() if estimator.img_pool_points is not None else []
            }
            
            json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
            json_path = os.path.join(args.data_output, json_filename)
            
            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            with open(json_path, "w") as f:
                json.dump(homography_data, f, indent=4)

        annotated_frame = estimator.visualize_corners(frame)
        
        output_path = os.path.join(args.output, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_frame)
        print(f"\tProcessed {image_path}")

    print("Homography estimation completed.")