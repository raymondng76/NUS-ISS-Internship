# NUS-ISS-Internship 2020

## Project Title:
---
Multi Object Multi Camera Tracking using ReID for traffic surveillance

## Requirements:
---
The main MultiVideoProcessor only requires the following dependencies:
1. Python >= 3.6
2. Pytorch >= 1.0

For all detector/tracker/ReID algorithm, please refer to their respective Github page for their requirements.

## Usage Guide:
---
1. Open config.ini and update entries accordingly, key entries to take note of are as follows:
 * query_video
 * gallery_video_folder
 * det_algorithm
 * reid_algorithm
 * Ensure all weights and model path are correct for their respective algorithm

2. Activate conda environment
3. Run the MultiVideoProcessor with the follow command
```
python MultiVideoProcessor.py
```

## Add new detector / tracker / ReID algorithm:
---
Please refer to [add_algorithm.md](https://github.com/raymondng76/NUS-ISS-Internship/blob/master/add_algorithm.md) for detailed step by step guide.