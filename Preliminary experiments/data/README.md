## Data directory
The input and output files used and produced by the A1, A2, and A3 scripts are collectively stored in the data directory.

### 📁 1_openpose-keypoints
The localised key points produced by OpenPose is made available in this directory. Only a subset of the original pose dataset is provided due to Github storage size restrictions. The full dataset can however be manually downloaded from [NWU Dayta Repository](https://doi.org/10.25388/nwu.23290937) and placed in this directory.

### 📁 2_generated-data
Each of the generated datasets that correspond to a specific augmentation scheme is output to this directory by script A1 as balanced datasets.
The script produces the following augmentation-based datasets:
  * baseline
  * dot
  * dot_blend
  * dot_blend_conf
  * cross
  * cross_blend
  * cross_blend_conf

### 📁 3_CNN-exploration
Both the trained models and associated logs produced by script A2 are stored in this directory as part of the multiple CNN configuration search to identify an optimal CNN architecture. 

### 📁 4_CNN-training-results
Both the trained models and associated logs produced by script A3 are stored in this directory which employs the optimal configuration identified in script A2.
