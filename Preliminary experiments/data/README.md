## Data directory
The input and output files used and produced by the B1 and B2 scripts are collectively stored in the data directory.

### 📁 1_video-frames
Only a subset of the original fall dataset is provided in this directory to allow for demonstratation of the data generation and model training process. The full dataset is not provided due to Github storage size restrictions. The full dataset can however be manually downloaded from [www.falldataset.com](https://www.falldataset.com/) and placed in this directory.

### 📁 2_openpose-keypoints
The localised key points produced by OpenPose for both the corresponding subset of the fall dataset and the complete dataset is made available in this directory.

### 📁 3_generated-data
Each of the generated datasets that correspond to a specific augmentation scheme are output to this directory by script B1. The script produces the following augmentation-based datasets:
  * 0_Baseline
  * 1_JointColour (Each key point is assigned a distinct hue)
  * 2_RadialGradient
  * 3_RadialSegment
  * 4_RingGradient
  * 5_RingSegment

### 📁 4_CNN-training-results
Both the trained models and associated logs produced by script B2 are stored in this directory.
