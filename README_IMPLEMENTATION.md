# CARLA Pose Estimation Data Generator

Enhanced CARLA simulation script for generating pose estimation training data with automatic weather variation.

## Features

### ✨ New in This Version

1. **YOLO-Pose Format Output** 
   - Normalized coordinates (0-1 range)
   - Bounding boxes computed from visible keypoints
   - 16 CARLA keypoints per person
   - Visibility flags for each keypoint

2. **Automatic Weather Reset**
   - Every 5 minutes, simulation resets with new random weather
   - 7 weather templates: sunny, cloudy, rainy, foggy, twilight, night, storm
   - Random variations added to each template
   - Full actor respawn (vehicles and pedestrians)

3. **Code Quality Improvements**
   - Type hints on all functions
   - Comprehensive docstrings
   - PEP 8 compliant
   - Modular, reusable functions
   - Better error handling with logging

## Output Format

### YOLO-Pose Annotation Format
Each line in annotation files (`Annot/*.txt`) represents one person:

```
<class> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_vis> ... <kp16_x> <kp16_y> <kp16_vis>
```

Where:
- `class`: Always 0 (person)
- `x_center, y_center, width, height`: Normalized bounding box [0, 1]
- `kpN_x, kpN_y`: Normalized keypoint coordinates [0, 1]
- `kpN_vis`: Visibility flag (0=not visible, 2=visible)

### 16 CARLA Keypoints
1. Head
2. Right Eye
3. Left Eye
4. Right Shoulder
5. Left Shoulder
6. Right Arm (Elbow)
7. Left Arm (Elbow)
8. Right Forearm (Wrist)
9. Left Forearm (Wrist)
10. Hip Center
11. Right Thigh (Hip)
12. Left Thigh (Hip)
13. Right Leg (Knee)
14. Left Leg (Knee)
15. Right Foot (Ankle)
16. Left Foot (Ankle)

## Directory Structure

```
{OUT_DIR}/
├── events/         # DVS event camera images (PNG)
├── RGB/           # RGB camera images (PNG)
├── GT/            # Ground truth skeleton visualizations (PNG)
└── Annot/         # YOLO-pose format annotations (TXT)
```

## Installation

### Requirements
- Python 3.7+
- CARLA 0.9.15
- NumPy 1.23.5
- OpenCV 4.8.1

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# CARLA egg file should be present (already included)
# carla-0.9.15-py3.7-linux-x86_64.egg
```

## Usage

### Basic Usage

```bash
python GenerateData.py
```

### Common Options

```bash
# Specify number of vehicles and pedestrians
python GenerateData.py -n 50 -w 20

# Use specific seed for reproducibility
python GenerateData.py --seed 42 --seedw 100

# Enable car lights
python GenerateData.py --car-lights-on

# Run in asynchronous mode
python GenerateData.py --asynch
```

### All Command Line Arguments

```
--map_name MAP        Map to load (default: Town10HD)
--host H              IP of CARLA server (default: 127.0.0.1)
-p, --port P          TCP port (default: 2000)
-n N                  Number of vehicles (default: 30)
-w W                  Number of walkers (default: 10)
--safe                Avoid spawning accident-prone vehicles
--filterv PATTERN     Vehicle model filter (default: "vehicle.*")
--generationv G       Vehicle generation: "1", "2", or "All"
--filterw PATTERN     Pedestrian filter (default: "walker.pedestrian.*")
--generationw G       Pedestrian generation (default: "2")
--tm-port P           Traffic Manager port (default: 8000)
--asynch              Asynchronous mode
--hybrid              Hybrid physics mode for Traffic Manager
-s, --seed S          Random seed for deterministic mode
--seedw S             Seed for pedestrians (default: 0)
--car-lights-on       Enable automatic car light management
--hero                Set one vehicle as hero
--respawn             Auto-respawn dormant vehicles (large maps)
--no-rendering        Disable rendering mode
```

## Weather Reset System

The simulation automatically resets every 5 minutes (300 seconds):

1. All actors (vehicles, pedestrians, cameras) are destroyed
2. New random weather is generated from templates
3. All actors respawn at new locations
4. Data generation continues seamlessly

This creates diverse training data with varied weather conditions.

## Validation

Use the included validation script to check annotation format:

```bash
python validate_yolo_pose.py /path/to/Annot/0001.txt
```

The script validates:
- All coordinates are in [0, 1] range
- Format matches YOLO-pose specification (53 values per line)
- Bounding boxes are valid
- All 16 keypoints are present
- Visibility flags are correct (0 or 2)

## Key Functions

### Helper Functions
- `check_keypoint_visibility()` - Detect if keypoint is in frame
- `compute_bbox_from_keypoints()` - Calculate bounding box from visible keypoints
- `generate_random_weather()` - Create random weather from templates

### Spawning Functions
- `spawn_vehicles()` - Spawn vehicles with autopilot
- `spawn_walkers()` - Spawn pedestrians with AI controllers
- `spawn_cameras()` - Spawn DVS and RGB cameras

### Processing Functions
- `GenerateGTPose()` - Generate YOLO-pose format annotations
- `ProcessDVSImage()` - Save DVS event images
- `ProcessRGBImage()` - Save RGB camera images

### Utility Functions
- `destroy_actors()` - Clean up all simulation actors
- `getCamXforms()` - Get camera transforms for specific maps
- `build_projection_matrix()` - Build camera intrinsic matrix

## Configuration

Edit these constants in the script to adjust behavior:

```python
BBOX_PADDING = 0.1              # Padding around bounding boxes (10%)
WEATHER_RESET_INTERVAL = 300     # Weather reset interval (seconds)
OUT_DIR = "/path/to/output"      # Output directory
MAP_NAME = 'Town10HD'            # CARLA map name
```

## Output File Naming

- DVS Events: `{frame_number}.png`
- RGB Images: `{frame_number}_RGB.png`
- Ground Truth: `{frame_number}.png`
- Annotations: `{frame_number}.txt`

Frame numbers are synchronized across all output types.

## Performance Tips

1. **Synchronous Mode** (default): More stable, recommended for data generation
2. **Hybrid Physics**: Use `--hybrid` for better performance with many actors
3. **No Rendering**: Use `--no-rendering` to speed up generation (no spectator view)
4. **Fewer Actors**: Reduce `-n` and `-w` if experiencing slowdowns

## Troubleshooting

### Common Issues

1. **"Connection refused"**: Start CARLA server first
   ```bash
   ./CarlaUE4.sh -quality-level=Low
   ```

2. **Memory issues during reset**: Normal - actors are destroyed and respawned
   - Monitor with: `watch -n 1 nvidia-smi` (GPU memory)

3. **Missing annotations**: Pedestrians may be out of camera view
   - Check camera position with spectator view
   - Adjust spawn locations or camera angles

4. **Invalid coordinates**: Run validation script to diagnose
   ```bash
   python validate_yolo_pose.py Annot/0001.txt
   ```

## Known Limitations

- Keypoint occlusion detection is basic (bounds checking only)
- Could be enhanced with depth camera or semantic segmentation
- Currently uses visibility flags 0 and 2 (not 1 for occluded)

## Future Enhancements

- [ ] Add depth/semantic camera for better occlusion detection
- [ ] Make weather reset interval configurable via CLI
- [ ] Add option to continue frame numbering across resets
- [ ] Save reset statistics (weather types, timestamps)
- [ ] Add validation mode (dry run without saving)

## References

- CARLA Documentation: https://carla.readthedocs.io/en/0.9.15/
- YOLO-Pose: https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose
- COCO Keypoints: https://cocodataset.org/#keypoints-2020

## License

This work is licensed under the terms of the MIT license.

## Citation

If you use this script in your research, please cite:

```bibtex
@misc{carla_pose_generator,
  title={CARLA Pose Estimation Data Generator},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourrepo}}
}
```
