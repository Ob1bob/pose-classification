import os
import math
import random
import colorsys
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot
from options import data_generate_options


def create_directory(path):
	""" Create directories in the path if they don't exist """
	if not os.path.exists(path):
		os.makedirs(path)


def get_spaced_hex_colours(n):
	""" Returns a list of evenly spaced hexadecimal color codes based on the HSV color space """
	hsv_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
	hex_out = []
	for rgb in hsv_tuples:
		rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
		hex_out.append('#%02x%02x%02x' % tuple(rgb))
	return hex_out


def hex_to_rgb(hex):
	""" Converts a hexadecimal color code to an RGB tuple """
	hex = hex.lstrip('#')
	h_length = len(hex)
	return tuple(int(hex[i:i + h_length // 3], 16) for i in range(0, h_length, h_length // 3))


def calculate_lightness(lightness, r, g, b):
	""" Applies a lightness transformation to an RGB color """
	lightness = (lightness - 1)  # reverse confidence
	lightness = ((lightness - 0) / (0.1 - 0)) * (0.02 - 0.1) + 0  # normalise
	# l = l/100 # turn into percentage

	tmp_r = (255 - r) * lightness + r
	tmp_g = (255 - g) * lightness + g
	tmp_b = (255 - b) * lightness + b

	# If a temporary value exceeds 255 (maximum value for an RGB color channel),
	# it is reverted back to the original color channel value to prevent color distortion.
	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def mix_two_rgb(r1, g1, b1, r2, g2, b2):
	""" Calculates the average of two RGB colors """
	tmp_r = (r1 * 0.5) + (r2 * 0.5)
	tmp_g = (g1 * 0.5) + (g2 * 0.5)
	tmp_b = (b1 * 0.5) + (b2 * 0.5)

	# If a temporary value exceeds 255 (maximum value for an RGB color channel),
	# it is reverted back to the original color channel value to prevent color distortion.
	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def get_dot_coordinates(x, y):
	""" Returns the x, y coordinate as a tuple """
	arr_out = [(x, y)]
	return arr_out


def get_diamond_coordinates(x, y, d):
	""" Generates a set of coordinates forming a diamond shape around a central point

	Arguments:
		x, y : coordinates of the central point
		d 	 : distance (size) of the diamond
	"""
	arr_out = []

	# create diamond shape
	shift = 0
	arr = np.arange(d + 1)
	for i in reversed(arr):
		shift += 1

		# right
		if 0 < x + i < opt.x_axis:
			arr_out.append((x + i, y))
			# up
			for j in range(x, x + i):
				if 0 < y + shift < opt.y_axis:
					arr_out.append((j, y + shift))
			# down
			for j in range(x, x + i):
				if 0 < y + shift < opt.y_axis:
					arr_out.append((j, y - shift))

		# left
		if 0 < x - i < opt.x_axis:
			arr_out.append((x - i, y))
			# up
			for j in range(x, x - i, -1):
				if 0 < y + shift < opt.y_axis:
					arr_out.append((j, y + shift))
			# down
			for j in range(x, x - i, -1):
				if 0 < y - shift < opt.y_axis:
					arr_out.append((j, y - shift))

	# removes any duplicate by converting list to a set and then back to list
	arr_out = list(set(arr_out))
	return arr_out


def get_crosshair_coordinates(x, y, d):
	""" Generates a set of coordinates forming a crosshair shape around a central point

	Arguments:
		x, y : coordinates of the central point
		d 	 : distance (size) of the crosshair
	"""
	arr_out = []

	# create crosshair shape
	stride = 0
	arr = np.arange(d + 1)
	for i in reversed(arr):
		stride += 1

		# right
		if 0 < x + i < opt.x_axis:
			arr_out.append((x + i, y))

		# left
		if 0 < x - i < opt.x_axis:
			arr_out.append((x - i, y))

		# up
		if 0 < y + i < opt.y_axis:
			arr_out.append((j, y + i))

		# down
		if 0 < y + i < opt.y_axis:
			arr_out.append((j, y - i))

	arr_out = list(set(arr_out))
	return arr_out


def balance_data_set(df0, sit_indexes, stand_indexes, upper_limit):
	""" Balances a data set by selecting an equal number of samples from the "sit" and "stand" classes """
	sit_index = sit_indexes[:upper_limit]  # restrict the number of samples to the same sample size
	stand_index = stand_indexes[:upper_limit]
	# throw indexes together that refer to the samples that will be used
	selected_samples_idx = sit_index + stand_index
	selected_samples_idx = sorted(selected_samples_idx, key=int)
	# restrict the imported data to the selected samples based on their index in the dataframe
	df2 = df0.iloc[selected_samples_idx, :]
	# randomise the order of rows in the DataFrame
	df2 = df2.sample(frac=1).reset_index(drop=True)
	df2 = df2.sample(frac=1).reset_index(drop=True)
	df2 = df2.sample(frac=1).reset_index(drop=True)
	return df2


def plot_image(img, title):
	""" Simple plot image function that takes a 3D array of RGB pixel values """
	pyplot.figure(figsize=(4, 4))
	pyplot.title(title)
	pyplot.imshow(img)
	pyplot.show()


def divide_data_by_class(df):
	""" Compile two arrays that contain the indexes of sitting or standing instances """
	cl = df[[' class (number)']]
	sit_indexes = []
	stand_indexes = []
	for index, cls in enumerate(cl.values):
		if cls == 0:  # sit is class 0; stand is class 1
			sit_indexes.append(index)
		else:
			stand_indexes.append(index)

	return sit_indexes, stand_indexes


if __name__ == "__main__":
	opt = data_generate_options()

	# declare output directory paths
	output_dir = os.path.join(opt.output, opt.dataset_name)
	output_dir_samples = os.path.join(opt.output, opt.dataset_name, "image_samples")
	# create output directory paths
	create_directory(output_dir)
	create_directory(output_dir_samples)

	# user feedback
	print("Generating data set: " + opt.dataset_name)
	print("Data set output location: " + output_dir)

	# create three separate 2D arrays complete with column names & normalise coordinates
	print("Reading key point coordinates from: " + opt.input)
	df_openpose = pd.read_csv(opt.input)  # import data from CSV file
	df_openpose = df_openpose.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe

	# report the proportions of sample representation in the data set
	sit_indexes, stand_indexes = divide_data_by_class(df_openpose)
	print("Sit sample size:\t{}".format(len(sit_indexes)))
	print("Stand sample size:\t{}".format(len(stand_indexes)))

	# limit & balance the data set to allow equal representation of sit and stand samples
	print("Re-balancing the data set according class distribution...")
	df_balanced = balance_data_set(df_openpose, sit_indexes, stand_indexes, opt.limit)
	# divide data again by class after re-balancing
	sit_indexes, stand_indexes = divide_data_by_class(df_balanced)
	print("Sit sample size:\t{}".format(len(sit_indexes)))
	print("Stand sample size:\t{}".format(len(stand_indexes)))

	# extract data from balanced dataframe
	# x_coordinates
	x = df_balanced.columns[df_balanced.columns.str.startswith('x')]
	x_cor = df_balanced[x].round(6)
	x = df_balanced.filter(regex='x\d+')
	x_cor = x.round(6)
	# y_coordinates
	y = df_balanced.columns[df_balanced.columns.str.startswith('y')]
	y_cor = df_balanced[y].round(6)
	y = df_balanced.filter(regex='y\d+')
	y_cor = y.round(6)
	# c_coordinates (confidence of coordinate)
	c = df_balanced.filter(regex='c\d+')
	c_cor = c.round(6)
	# class of each image
	cl = df_balanced[[' class (number)']]

	# adjust the x and y coordinates in relation to the height and width of the image size (pixels)
	highest_x = x_cor.values.max()
	highest_y = y_cor.values.max()
	x_cor = ((x_cor - 0) / (highest_x - 0) * (opt.x_axis - 1)).round(0)
	y_cor = ((y_cor - 0) / (highest_y - 0) * (opt.y_axis - 1)).round(0)

	# create RGB colours
	spacedRGB = []
	spacedHex = get_spaced_hex_colours(len(x_cor.columns))  # number of key points (body joints) = x coordinate df columns
	for hex in spacedHex:
		rgb = hex_to_rgb(hex)
		spacedRGB.append(rgb)

	# generate pose data images and store them in lists within a dictionary
	# declare dictionary with preset keys and null values
	keys = ['baseline', 'dot', 'dot_conf', 'dot_conf_blended', 'cross', 'cross_conf', 'cross_conf_blended']
	dict_image_lists = {key: [] for key in keys}

	# colour code and map the key points/body joints to a generated image
	for key in dict_image_lists:
		print("Generating image set:\t " + key)

		img_set = []
		max_samples = 50  # limit the number of samples to generate
		sample_subdir = os.path.join(output_dir_samples, key)
		create_directory(sample_subdir)

		for row in range(x_cor.shape[0]):
			image = np.full((opt.x_axis, opt.y_axis, 3), 0)  # 3D array with black BG: 255 (white) / 0 (black)

			for col in range(x_cor.shape[1]):
				x = int(x_cor.iloc[row, col])
				y = int(y_cor.iloc[row, col])
				confidence = c_cor.iloc[row, col]
				r, g, b = spacedRGB[col]
				if "baseline" in key:
					r, g, b = 255, 255, 255  # white pixel

				if "conf" in key:
					r, g, b = calculate_lightness(confidence, r, g, b)

				if not (x == 0 and y == 0 and confidence == 0):  # skip non-localised key points
					if "dot" in key or "baseline" in key:
						plot_coor = get_dot_coordinates(x, y)
					else:
						plot_coor = get_diamond_coordinates(x, y, 1)
					for x, y in plot_coor:
						r0, g0, b0 = image[y, x]
						if (r0 == g0 == b0 == 0) or (r0 == g0 == b0 == 255):
							image[y, x] = (r, g, b)  # swap x & y coordinates (Python represents 3D array differently)
						elif "blended" in key:
							image[y, x] = mix_two_rgb(r, g, b, r0, g0, b0)

			# Use PIL to create an image from the array of pixels
			if opt.create_samples and row < max_samples:
				array = np.array(image, dtype=np.uint8)
				new_image = Image.fromarray(array)
				new_image.save(os.path.join(sample_subdir, str(cl.values[row][0]) + "_" + df_balanced.iloc[row, 0]))  # filename

			# store image in img_set
			img_set.append(image)
		# store the image set in dictionary
		dict_image_lists[key].append(img_set)

	print("All key point images generated.\n")

	# divide img sets into training and test sets and write to file
	# Iterate over the dictionary to obtain each array under a specific key
	for key, img_set_list in dict_image_lists.items():
		# iterte over image_set to overwrite train & test sets with image along with its class which is in order of the image_set
		sit_index = []
		stand_index = []
		for array_list in img_set_list:
			for index, array in enumerate(array_list):
				if cl.values[index][0] == 1:
					stand_index.append(index)
				else:
					sit_index.append(index)
			print("{} data set size:\n"
						 "\tSit samples:\t{}\n\tStand samples:\t{}".format(key, len(sit_index), len(stand_index)))

			# identify the index at which data is to be split between train and test sets
			split_anchor_idx = round(len(sit_index) * 0.8)

			train_index = sit_index[:split_anchor_idx] + stand_index[:split_anchor_idx]
			test_index = sit_index[split_anchor_idx:] + stand_index[split_anchor_idx:]
			random.shuffle(train_index)
			random.shuffle(test_index)

			# debug
			for index in test_index:
				plot_image(array_list[index], str(cl.values[index]))
			for index in train_index:
				plot_image(array_list[index], str(cl.values[index]))

			# convert train & test data sets to 1D arrays and write them to file
			train_set = []
			for index in train_index:
				# position 0 is class; remainder is the flattened 3D array
				train_set.append(np.concatenate([cl.values[index], np.concatenate(array_list[index]).ravel()]))
			test_set = []
			for index in test_index:
				test_set.append(np.concatenate([cl.values[index], np.concatenate(img_set_list[index]).ravel()]))
			# write the arrays to file
			csv_out_train = os.path.join(output_dir, key + "-TRAIN.csv")
			csv_out_test = os.path.join(output_dir, key + "-TEST.csv")
			pd.DataFrame(train_set).to_csv(csv_out_train, index=False)
			pd.DataFrame(test_set).to_csv(csv_out_test, index=False)

			print("{} data set size: {}\n"
				  "\tTrain set:\t{}\n\tTest set:\t{}\n".
				  format(key, len(train_set) + len(test_set), len(train_set), len(test_set)))

	print("DONE. All key point data sets written to file.\n")