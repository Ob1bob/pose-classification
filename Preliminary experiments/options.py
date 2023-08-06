
import argparse
import os

cwd = os.getcwd()

def data_generate_options():
    parser = argparse.ArgumentParser(description='Arguments for creating data set in the preliminary experiments')
    parser.add_argument("-n", "--name", default='pose_set', dest='dataset_name', type=str, help='name of the data set')
    parser.add_argument("-x", "--x_axis", default=32, type=int, help='pixel size of image vertical axis')
    parser.add_argument("-y", "--y_axis", default=32, type=int, help='pixel size of image horizontal axis')
    parser.add_argument("-s", "--samples", default=True, dest='create_samples', type=bool, help='generate sample images')
    parser.add_argument("-l", "--limit", default=12500, type=int, help='limit the class sample size')

    parser.add_argument("-i", "--input", default=os.path.join(cwd, "openpose-keypoints", "subset_AnnexureSamples.csv"), type=str,
                        help='path to csv with OpenPose key points')
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "generated-data"), type=str,
                        help='path to directory where generated pose set is to be stored')
    opt = parser.parse_args()
    return opt


# class variables
cwd = os.getcwd()
# directories & files
dataset_name = "blackBG_cross_blended"  # CHANGE
dataset_size = "Small" # or Small
dataset_dir = os.path.join(cwd, "Generated/Datasets/"+dataset_size)  # CHANGE
parent_dir = os.path.join(cwd, "CNN\\Best_Architecture\\"+dataset_name+"-"+dataset_size)  # CHANGE
# subdirectories
MODEL_DIR = os.path.join(parent_dir, "Models")
LOG_DIR = os.path.join(parent_dir, "logs")

# explore 32 different CNN architectures
def exploration_options():
    parser = argparse.ArgumentParser(description='Arguments for training a CNN in the preliminary set of experiments')
    parser.add_argument("-b", "--batch", default=32, dest='batch_size', type=int, help='size of sample training batches')
    parser.add_argument("-e", "--epochs", default=32, type=int, help='number of training epochs')

    parser.add_argument("-l", "--log", default=os.path.join(cwd, "NN"), dest='log_dir', type=str,
                        help='path to directory where logs are to be stored')
    parser.add_argument("-d", "--data", default=os.path.join(cwd, "data", "pose_set"), dest='data_dir', type=str,
                        help='path to directory where generated pose set is to be read from for training and testing')
    # architecture exploratory experimentation was conducted using baseline and cross_conf_blended data sets
    parser.add_argument("-n", "--name", default="baseline", dest='dataset_name', type=str,
                        help='name of data set to use for training and testing')
    opt = parser.parse_args()
    return opt

# explore 32 different CNN architectures
def training_options():
    parser = argparse.ArgumentParser(description='Arguments for training a CNN in the preliminary set of experiments')