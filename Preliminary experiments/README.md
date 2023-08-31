# Preliminary experiments
The preliminary experiments are an initial investigation into how data augmentation as a form of feature engineering can be purposed for pose classification. These experiments are intended to act as a proof of concept to establish the viability of novel augmentation techniques for pose expression.

### <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" width="20" height="20" border="10"/> script.A1.DataGeneration.py
The 18 localised key points and confidence scores predicated using OpenPose are used to generate a image dataset. The data generation script transforms the XY-coordinates into a collection of seven different datasets that each incorporates a different form of data augmentation. All of the sets depict the same OpenPose joint mappings but in different formats within 32x32 image frame.

<p align="center">
<img src="https://github.com/dulocian/pose-classification/blob/main/images/A1-Sample.png"/>
</p>


### <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" width="20" height="20" border="10"/> script.A2.CNNExploration.py
A non-exhaustive search is conducted to identify an optimal CNN architecture conducive to classifying the generated pose data. The aim is to ensure a fair evaluation by selecting a CNN layer and hyperparameter configuration that does not favour a particular augmentation over another. The search space consists of 32 different architectures derived from all possible combinations of standard hyperparameter specifications listed in the Table below. 

#### CNN hyperparameter configuration values
| Network component	| Configuration |
| ---------- | :--------- |
| Number of convolutional layers | [1, 2, 3, 4] layers |
| Number of trainable convolutional filters  | [32, 64, 128, 256] filters  |
| Number of fully connected layers  | [1, 2] layers   |
| Number of nodes in hidden fully connected layer | [32, 64, 128, 256] nodes |
| Number of nodes in output fully connected layer | 2 nodes |
| Batch size | 32 |
| Normalisation technique | Batch normalisation |	
| Activation function | Rectified linear unit (ReLU) |


### <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" width="20" height="20" border="10"/>  script.A3.CNNPoseClassifier.py
The best performing architecture determined in script A2 is the 2:conv-64:nodes-2:dense CNN architecture. This CNN architecture consists of two convolutional neural layers (2:conv), 64 trainable convolutional filters (64:nodes), and two additional fully connected layers (2:dense) preceding the final dense layer. Script A3 is similar to A2 except that is purposed to only train a model using the described network configuration.

<p align="center">
<img src="https://github.com/dulocian/pose-classification/blob/main/images/A3-CNN.png"/>
</p>

