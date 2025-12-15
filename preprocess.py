import cv2
import os
import shutil
import numpy as np
import argparse

from typing import List, Tuple, Optional


def is_pool_frame(frame: np.ndarray, blue_threshold: float = 0.15) -> bool:
    """
    Detect if frame contains a water pool based on blue color prevalence
    
    Args:
        frame: Input image (BGR)
        blue_threshold: Minimum ratio of blue pixels to consider as pool frame
        
    Returns:
        True if frame likely contains a pool
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for blue/cyan colors (typical pool water)
    # Hue: 90-130 (blue/cyan), Saturation: 30-255, Value: 30-255
    lower_blue = np.array([90, 30, 30])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Calculate ratio of blue pixels
    blue_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    
    return blue_ratio >= blue_threshold


def extract_frames_by_count(video_path: str, 
                            num_frames: int = 10,
                            output_dir: str = "extracted_frames",
                            check_pool: bool = True,
                            skip_similar: bool = True,
                            similarity_threshold: float = 0.95) -> List[str]:
    """
    Extract a specified number of frames evenly distributed throughout the video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        output_dir: Directory to save extracted frames
        check_pool: Whether to check if frame contains pool
        skip_similar: Skip frames that are too similar to previous
        similarity_threshold: Similarity threshold (0-1) for skipping frames
        
    Returns:
        List of paths to extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame indices to extract
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    extracted_paths = []
    prev_frame = None
    pool_skipped = 0
    similar_skipped = 0
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Check if frame contains pool
        if check_pool and not is_pool_frame(frame):
            pool_skipped += 1
            continue
        
        # Check similarity with previous frame
        if skip_similar and prev_frame is not None:
            similarity = compare_frames(frame, prev_frame)
            if similarity > similarity_threshold:
                similar_skipped += 1
                continue
        
        # Save frame
        output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_paths.append(output_path)
        prev_frame = frame.copy()
        
        print(f"Extracted frame {len(extracted_paths)}/{num_frames}: {output_path}")
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  - Extracted: {len(extracted_paths)} frames")
    print(f"  - Skipped (non-pool): {pool_skipped}")
    print(f"  - Skipped (similar): {similar_skipped}")
    
    return extracted_paths


def extract_frames_by_interval(video_path: str,
                               interval_seconds: float = 1.0,
                               max_frames: Optional[int] = None,
                               output_dir: str = "extracted_frames",
                               check_pool: bool = True,
                               skip_similar: bool = True,
                               similarity_threshold: float = 0.95) -> List[str]:
    """
    Extract frames at regular time intervals
    
    Args:
        video_path: Path to video file
        interval_seconds: Time interval between frames (in seconds)
        max_frames: Maximum number of frames to extract (None for unlimited)
        output_dir: Directory to save extracted frames
        check_pool: Whether to check if frame contains pool
        skip_similar: Skip frames that are too similar to previous
        similarity_threshold: Similarity threshold (0-1) for skipping frames
        
    Returns:
        List of paths to extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    print(f"Extracting frames every {interval_seconds}s")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
        frame_interval = 1
    
    extracted_paths = []
    prev_frame = None
    pool_skipped = 0
    similar_skipped = 0
    frame_idx = 0
    
    while True:
        if max_frames and len(extracted_paths) >= max_frames:
            print(f"\nReached maximum frame limit: {max_frames}")
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if frame contains pool
        if check_pool and not is_pool_frame(frame):
            pool_skipped += 1
            frame_idx += frame_interval
            continue
        
        # Check similarity with previous frame
        if skip_similar and prev_frame is not None:
            similarity = compare_frames(frame, prev_frame)
            if similarity > similarity_threshold:
                similar_skipped += 1
                frame_idx += frame_interval
                continue
        
        # Save frame
        timestamp = frame_idx / fps
        output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_t{timestamp:.2f}s.jpg")
        cv2.imwrite(output_path, frame)
        extracted_paths.append(output_path)
        prev_frame = frame.copy()
        
        print(f"Extracted frame {len(extracted_paths)} at {timestamp:.2f}s: {output_path}")
        
        frame_idx += frame_interval
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  - Extracted: {len(extracted_paths)} frames")
    print(f"  - Skipped (non-pool): {pool_skipped}")
    print(f"  - Skipped (similar): {similar_skipped}")
    
    return extracted_paths


def compare_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compare similarity between two frames using histogram correlation
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Similarity score (0-1, higher means more similar)
    """
    # Resize frames to same size if needed
    if frame1.shape != frame2.shape:
        h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation


def extract_calibration_frame(video_path: str, 
                              frame_number: int = 100, 
                              output_path: str = None) -> str:
    """
    Extract a single frame from video for homography calibration
    
    Args:
        video_path: Path to video file
        frame_number: Which frame to extract
        output_path: Where to save the frame
        
    Returns:
        Path to saved frame
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    # Skip to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return None
    
    if output_path is None:
        output_path = f"calibration_frame_{frame_number}.jpg"
    
    cv2.imwrite(output_path, frame)
    print(f"Saved calibration frame to {output_path}")
    return output_path


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    info = {
        'path': video_path,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    
    return info


def clean_data_dir(data_dir: str = "data", keep_dirs: List[str] = ["video"]):
    """
    Clean all directories in data directory except specified ones
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    print(f"Cleaning {data_dir}...")
    for item in os.listdir(data_dir):
        if item in keep_dirs:
            continue
            
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"  Removed directory: {item}")
        else:
            os.remove(item_path)
            print(f"  Removed file: {item}")
    print("Clean complete.")



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Extract frames from water polo video")
    parser.add_argument("video_path", nargs="?", help="Path to video file (required for extraction modes)")
    parser.add_argument("--mode", choices=["count", "interval", "info", "clean"], default="count",
                       help="Extraction mode: count, interval, info, or clean")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of frames to extract (count mode)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Time interval in seconds (interval mode)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to extract (interval mode)")
    parser.add_argument("--output-dir", default="data/frames",
                       help="Output directory for frames")
    parser.add_argument("--no-pool-check", action="store_true",
                       help="Disable pool detection")
    parser.add_argument("--no-skip-similar", action="store_true",
                       help="Disable skipping similar frames")
    parser.add_argument("--similarity", type=float, default=0.95,
                       help="Similarity threshold (0-1)")
    
    args = parser.parse_args()
    
    if args.mode == "clean":
        clean_data_dir()
        
    elif args.mode == "info":
        if not args.video_path:
            parser.error("video_path is required for info mode")
        info = get_video_info(args.video_path)
        if info:
            print("\n=== Video Information ===")
            print(f"Path: {info['path']}")
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"Frame Count: {info['frame_count']}")
            print(f"FPS: {info['fps']:.2f}")
            print(f"Duration: {info['duration']:.2f} seconds")
    
    elif args.mode == "count":
        if not args.video_path:
            parser.error("video_path is required for count mode")
        extract_frames_by_count(
            args.video_path,
            num_frames=args.count,
            output_dir=args.output_dir,
            check_pool=not args.no_pool_check,
            skip_similar=not args.no_skip_similar,
            similarity_threshold=args.similarity
        )
    
    elif args.mode == "interval":
        if not args.video_path:
            parser.error("video_path is required for interval mode")
        extract_frames_by_interval(
            args.video_path,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            check_pool=not args.no_pool_check,
            skip_similar=not args.no_skip_similar,
            similarity_threshold=args.similarity
        )
