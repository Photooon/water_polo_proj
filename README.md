# Water Polo Comp Vision Project

## General

- config.py: Centralizes pool dimensions, detection thresholds, tracking limits, and visualization defaults used across the project.
- preprocess.py: Utilities to sample frames from videos (by count or interval), filter out non-pool frames via color masking, and skip near-duplicate frames.
- homography.py: Segments water regions, extracts pool corners, computes a homography to a top-down pool plane, supports corner visualization and coordinate transforms.
- detector.py: YOLO-based player/ball detector with bounding-box extraction, center computation and visualization.
- pipeline.py: End-to-end pipeline that runs homography + detection, transforms player centroids to pool coordinates, and saves both annotated camera frames and top-down pool-view renders.
- utils.py: Helper functions.

**Before running, download the demo video from [Google Drive](https://drive.google.com/file/d/1xGRWWmyJcZfAnTErgTYfUiYyKPoL_wKX/view?usp=drive_link) and save it into "./data/video/demo.mp4".**
The original video is from [Youtube](https://www.youtube.com/watch?v=kgcabhpH968&t=1s).

## Usage

- Install

```bash
uv venv --python=3.12
uv pip install -r requirements.txt
```

- Extract image frames from video

```bash
python preprocess.py data/video/demo.mp4 --mode count --count 10
```

- Pool detection and Solve homography

```bash
python homography.py data/frames
```

For simplicity, solving homography requires two corners of the pool appear in the image, otherwise the result may be inaccurate.

- Players detection

```bash
python detector.py data/frames
```

- Heatmap

```bash
python heatmap.py 
```


- Full Pipeline (Homography + Detection) -> Topdown pool view

```bash
python pipeline.py data/frames
```

## TODO

- [x] Data Preprocessing
- [x] Fix Homography
- [x] Players Detection
- [ ] Players Track
- [ ] Analysis
  - [ ] HeatMap
  - [ ] Summary
- [ ] Final Report