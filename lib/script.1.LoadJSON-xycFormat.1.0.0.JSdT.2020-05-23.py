import numpy as np
import sys
import os
import cv2
import json
import pandas as pd
print('OpenCV - version: ', cv2.__version__)


# Load keypoint data from JSON output
column_names = ["Filename", " x0 (Nose)", " y0 (Nose)", " c0 (Nose)", " x1 (Neck)", " y1 (Neck)", " c1 (Neck)", " x2 (RShoulder)", " y2 (RShoulder)", " c2 (RShoulder)", " x3 (RElbow)", " y3 (RElbow)", " c3 (RElbow)", " x4 (RWrist)", " y4 (RWrist)", " c4 (RWrist)", " x5 (LShoulder)", " y5 (LShoulder)", " c5 (LShoulder)", " x6 (LElbow)", " y6 (LElbow)", " c6 (LElbow)", " x7 (LWrist)", " y7 (LWrist)", " c7 (LWrist)", " x8 (MidHip)", " y8 (MidHip)", " c8 (MidHip)", " x9 (RHip)", " y9 (RHip)", " c9 (RHip)", " x10 (RKnee)", " y10 (RKnee)", " c10 (RKnee)", " x11 (RAnkle)", " y11 (RAnkle)", " c11 (RAnkle)", " x12 (LHip)", " y12 (LHip)", " c12 (LHip)", " x13 (LKnee)", " y13 (LKnee)", " c13 (LKnee)", " x14 (LAnkle)", " y14 (LAnkle)", " c14 (LAnkle)", " x15 (REye)", " y15 (REye)", " c15 (REye)", " x16 (LEye)", " y16 (LEye)", " c16 (LEye)", " x17 (REar)", " y17 (REar)", " c17 (REar)", " x18 (LEar)", " y18 (LEar)", " c18 (LEar)", " x19 (LBigToe)", " y19 (LBigToe)", " c19 (LBigToe)", " x20 (LSmallToe)", " y20 (LSmallToe)", " c20 (LSmallToe)", " x21 (LHeel)", " y21 (LHeel)", " c21 (LHeel)", " x22 (RBigToe)", " y22 (RBigToe)", " c22 (RBigToe)", " x23 (RSmallToe)", " y23 (RSmallToe)", " c23 (RSmallToe)", " x24 (Rheel)", " y24 (Rheel)", " c24 (Rheel)", " class (number)", " class (text)"]

# Paths - should be the folder where Open Pose JSON output was stored
cur_dir = sys.path[0]
path_to_json = cur_dir+'/!DATA/2.OpenPoseCSV/160x120/JSON/'
path_to_json_parent = cur_dir+'/!DATA/2.OpenPoseCSV/160x120/'

# Loop through all json files in output directory
# Each file is a frame in the video
# If multiple people are detected - choose the most centered high confidence points
for j in set(["Training", "Validation", "Testing"]):
	f = open(path_to_json_parent + "/" + "Skipped_" + j + ".txt", "w+")
	path = path_to_json + "/" + j + "/"
	json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
	print('\nFound: ', len(json_files), 'json keypoint files in ' + j)

	compiled_set = []
	for i, file in enumerate(json_files):

		print("\tprocessing ("+str(i+1)+"/"+str(len(json_files))+"):\t", file)

		temp_df = json.load(open(path + file))
		temp = []

		keypoints = []
		if len(temp_df['people']) > 1:
			highest_conf_person = []
			for person in temp_df['people']:
				keypts = person['pose_keypoints_2d']
				avg_conf = 0
				for idx in range(2, 76, 3):
					avg_conf += keypts[idx]
				highest_conf_person.append(avg_conf / 25)
			max_index = np.argmax(np.array(highest_conf_person), axis=0)
			keypoints = temp_df['people'][max_index]['pose_keypoints_2d']
		elif len(temp_df['people']) == 1:
			keypoints = temp_df['people'][0]['pose_keypoints_2d']
		elif file.split('_')[4] == "6":
			keypoints = list(np.full(75, -1))
		else:
			f.write(file+"\n")
			continue

		keypoints.insert(0, file + ".png")
		keypoints.append(file.split('_')[4])
		keypoints.append(file.split('_')[3])

		# modify image array by prepending the class
		compiled_set.append(keypoints)

	csv_out = path_to_json_parent + "Keypoints_" + j + ".csv"

	# print
	out_df = pd.DataFrame(compiled_set)
	out_df.columns = column_names
	out_df.to_csv(csv_out, index=False)