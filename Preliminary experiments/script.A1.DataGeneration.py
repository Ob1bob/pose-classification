import os
import math
import random
import colorsys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from matplotlib import pyplot
from options import data_generate_options
import configparser


def create_directory(path):
	"""
	Create directories in the specified path does not exist.

	Args:
		path (str): Path of the directory to be created.
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def get_spaced_hex_colours(N):
	"""
	Generate visually distinct colours evenly distributed in the HSV colour space.

	Args:
		N (int): The number of distinct colours to generate.

	Returns:
		list: A list of hex colour codes representing visually distinct colours.
	"""
	HSV_tuples = [(x * 1.0 / N, 1.0, 1.0) for x in range(N)]
	hex_out = []
	for rgb in HSV_tuples:
		rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
		hex_out.append('#%02x%02x%02x' % tuple(rgb))
	return hex_out


def hex_to_rgb(hex):
	"""
	Convert a hexadecimal colour code into an RGB tuple,
	representing the corresponding red, green, and blue colour components.

	Args:
		hex_code (str): 	The hexadecimal colour code to be converted.

	Returns:
		tuple: An RGB tuple containing the red, green, and blue colour components.
	"""
	hex = hex.lstrip('#')
	h_length = len(hex)
	return tuple(int(hex[i:i + h_length // 3], 16) for i in range(0, h_length, h_length // 3))


def calculate_lightness(lightness, r, g, b):
	"""
	Apply a tint transformation to an RGB colour.

	Args:
		lightness (float): 	tint parameter (e.g., between 0 and 1).
		r (int): 			Red channel value (0-255).
		g (int): 			Green channel value (0-255).
		b (int): 			Blue channel value (0-255).

	Returns:
		tuple: Transformed RGB colour tuple.
	"""
	lightness = (lightness - 1)  # reverse confidence
	lightness = ((lightness - 0) / (0.1 - 0)) * (0.02 - 0.1) + 0  # normalise
	# l = l/100 # turn into percentage

	tmp_r = (255 - r) * lightness + r
	tmp_g = (255 - g) * lightness + g
	tmp_b = (255 - b) * lightness + b

	# If a temporary value exceeds 255 (maximum value for an RGB colour channel),
	# it is reverted back to the original colour channel value to prevent colour distortion.
	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def mix_two_rgb(r1, g1, b1, r2, g2, b2):
	"""
	Calculate the average of two RGB colours.

	Args:
		r1 (int): 	Red channel value of colour 1 (0-255).
		g1 (int): 	Green channel value of colour 1 (0-255).
		b1 (int): 	Blue channel value of colour 1 (0-255).
		r2 (int): 	Red channel value of colour 2 (0-255).
		g2 (int): 	Green channel value of colour 2 (0-255).
		b2 (int): 	Blue channel value of colour 2 (0-255).

	Returns:
		tuple: Average RGB colour tuple.
	"""
	tmp_r = (r1 * 0.5) + (r2 * 0.5)
	tmp_g = (g1 * 0.5) + (g2 * 0.5)
	tmp_b = (b1 * 0.5) + (b2 * 0.5)

	# If a temporary value exceeds 255 (maximum value for an RGB colour channel),
	# it is reverted back to the original colour channel value to prevent colour distortion.
	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def get_dot_coordinates(x, y):
	"""
	Returns the x, y coordinate as a tuple of dot format body key point to be mapped onto an image

	Args:
		x (int): 	The x-coordinate of the point.
		y (int): 	The y-coordinate of the point.

	Returns:
		tuple: A tuple containing the x and y coordinates.
	"""
	arr_out = [(x, y)]
	return arr_out


def get_diamond_coordinates(x, y, d):
	"""
	Generates a set of coordinates forming a diamond shape around a central point (x, y) with
	the given diamond diameter (d) as a body key point to be mapped onto an image

	Args:
		x (int): 	The x-coordinate of the central point.
		y (int): 	The y-coordinate of the central point.
		d (int): 	The diameter of the diamond (distance from the center to each corner).

	Returns:
		list: A list of coordinate tuples representing the generated diamond shape.
	"""
	arr_out = []

	# create diamond shape
	shift = 0
	arr = np.arange(d + 1)
	for i in reversed(arr):
		shift += 1

		# right
		if 0 < x + i < x_axis:
			arr_out.append((x + i, y))
			# up
			for j in range(x, x + i):
				if 0 < y + shift < y_axis:
					arr_out.append((j, y + shift))
			# down
			for j in range(x, x + i):
				if 0 < y + shift < y_axis:
					arr_out.append((j, y - shift))

		# left
		if 0 < x - i < x_axis:
			arr_out.append((x - i, y))
			# up
			for j in range(x, x - i, -1):
				if 0 < y + shift < y_axis:
					arr_out.append((j, y + shift))
			# down
			for j in range(x, x - i, -1):
				if 0 < y - shift < y_axis:
					arr_out.append((j, y - shift))

	# removes any duplicate by converting list to a set and then back to list
	arr_out = list(set(arr_out))
	return arr_out


def get_crosshair_coordinates(x, y, d):
	"""
	Generates a set of coordinates that form a cross-hair shape centered around the specified
	point (x, y) with the given diameter (d). as a body key point to be mapped onto an image.

	Args:
		x (int): 	The x-coordinate of the central point.
		y (int): 	The y-coordinate of the central point.
		d (int): 	The diameter of the cross-hair.

	Returns:
		list: A list of tuples representing the generated coordinates.
	"""
	arr_out = []

	# create crosshair shape
	stride = 0
	arr = np.arange(d + 1)
	for i in reversed(arr):
		stride += 1

		# right
		if 0 < x + i < x_axis:
			arr_out.append((x + i, y))

		# left
		if 0 < x - i < x_axis:
			arr_out.append((x - i, y))

		# up
		if 0 < y + i < y_axis:
			arr_out.append((j, y + i))

		# down
		if 0 < y + i < y_axis:
			arr_out.append((j, y - i))

	arr_out = list(set(arr_out))
	return arr_out


def balance_data_set(df0, sit_indexes, stand_indexes, upper_limit):
	"""
	Balances a data set by selecting an equal number of samples from the "sit" and "stand" classes.

	Args:
		df0 (pandas.DataFrame): 	The original DataFrame containing the data.
		sit_indexes (list): 		Indexes of samples from the "sit" class.
		stand_indexes (list): 		Indexes of samples from the "stand" class.
		upper_limit (int): 			The maximum number of samples to be selected from each class.

	Returns:
		pandas.DataFrame: The balanced DataFrame containing selected samples.
	"""
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


def plot_image(img, window_title, fig_title):
	"""
	Simple plot image function that takes a 3D array of RGB pixel values and displays it using matplotlib.

	Args:
		img (numpy.ndarray): 	The image array to be displayed.
		window_title (str): 	The title to be displayed in the window's title bar.
		fig_title (str): 		The title to be displayed above the image.
	"""
	pyplot.figure(figsize=(4, 4))
	pyplot.title(fig_title)
	pyplot.imshow(img)
	man = pyplot.get_current_fig_manager()
	man.canvas.set_window_title(window_title)
	# display plot
	pyplot.show()


def divide_data_by_class(df):
	"""
	Divide instances in a DataFrame into "sitting" and "standing" classes and return their indexes.

	Args:
		df (pandas.DataFrame): 	The DataFrame containing the data, with a "class (number)" column.

	Returns:
		sit_indexes (list): 	List of indexes corresponding to instances of the "sitting" class.
		stand_indexes (list): 	List of indexes corresponding to instances of the "standing" class.
	"""
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
	# read arguments from config file
	config = configparser.ConfigParser()
	config.read('parameters.conf')

	input_file = os.path.abspath(config.get('DataGen', 'input_file'))
	dataset_dir = os.path.abspath(config.get('Global', 'dataset_dir'))
	x_axis = int(config.get('DataGen', 'x_axis'))
	y_axis = int(config.get('DataGen', 'y_axis'))
	max_sample_images = int(config.get('DataGen', 'max_sample_images'))
	max_sample_dataset = int(config.get('DataGen', 'max_sample_dataset'))

	# declare output directory paths
	output_dir = os.path.join(dataset_dir, Path(input_file).stem)  # filename of input file is used as image output dir
	output_dir_samples = os.path.join(output_dir, "image_samples")
	# create output directory paths
	create_directory(output_dir)
	create_directory(output_dir_samples)

	# user feedback
	print("Generating data set: \n\t" + Path(input_file).stem)
	print("Data set output location: \n\t" + output_dir)

	# create three separate 2D arrays complete with column names & normalise coordinates
	print("Reading key point coordinates from: \n\t" + input_file)
	df_openpose = pd.read_csv(input_file)  # import data from CSV file
	df_openpose = df_openpose.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe

	# report the proportions of sample representation in the data set
	print("\nClass distribution:")
	sit_indexes, stand_indexes = divide_data_by_class(df_openpose)
	print("Sit sample size:\t{}".format(len(sit_indexes)))
	print("Stand sample size:\t{}".format(len(stand_indexes)))

	# limit & balance the data set to allow equal representation of sit and stand samples
	print("Re-balancing the data set according to class distribution...")
	df_balanced = balance_data_set(df_openpose, sit_indexes, stand_indexes, max_sample_dataset)
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
	image_filename = df_balanced[['Filename']]

	# adjust the x and y coordinates in relation to the height and width of the image size (pixels)
	highest_x = x_cor.values.max()  # can also specify highest value as 1.0 (openpose localisation output format)
	highest_y = y_cor.values.max()  # can also specify highest value as 1.0 (openpose localisation output format)
	#x_cor = ((x_cor - 0) / (highest_x - 0) * (x_axis - 1)).round(0)
	#y_cor = ((y_cor - 0) / (highest_y - 0) * (y_axis - 1)).round(0)
	x_cor = ((x_cor - 0) / (1 - 0) * (x_axis - 1)).round(0)
	y_cor = ((y_cor - 0) / (1 - 0) * (y_axis - 1)).round(0)

	# create RGB colours
	spacedRGB = []
	spacedHex = get_spaced_hex_colours(len(x_cor.columns))  # number of key points (body joints) = x coordinate df columns
	for hex in spacedHex:
		rgb = hex_to_rgb(hex)
		spacedRGB.append(rgb)

	# generate pose data images and store them in lists within a dictionary
	# declare dictionary with preset keys and null values
	keys = ['baseline', 'dot', 'dot_blend', 'dot_blend_conf', 'cross', 'cross_blend', 'cross_blend_conf']
	dict_image_lists = {key: [] for key in keys}

	# colour code and map the key points/body joints to a generated image
	for key in dict_image_lists:
		print("Generating image set:\t " + key)

		img_set = []
		sample_subdir = os.path.join(output_dir_samples, key)
		create_directory(sample_subdir)

		for row in range(x_cor.shape[0]):
			image = np.full((y_axis, x_axis, 3), 0)  # 3D array with black BG: 255 (white) / 0 (black)

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
					# plot the key point by colouring each individual pixel in its region
					for x, y in plot_coor:
						r0, g0, b0 = image[y, x]
						if (r0 == g0 == b0 == 0) or (r0 == g0 == b0 == 255):
							image[y, x] = (r, g, b)  # swap x & y coordinates (Python represents 3D array differently)
						elif "blend" in key:
							image[y, x] = mix_two_rgb(r, g, b, r0, g0, b0)

			# Use PIL to create an image from the array of pixels
			if row < max_sample_images:
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
		# iterate over image_set to overwrite train & test sets with image
		# along with its class which is in order of the image_set
		sit_index = []
		stand_index = []
		for array_list in img_set_list:
			for index, array in enumerate(array_list):
				if cl.values[index][0] == 1:
					stand_index.append(index)
				else:
					sit_index.append(index)
			print("[{}] data set size: {}\n"
						 "\tSit samples:\t{}\n\tStand samples:\t{}".format(key, (len(sit_index) + len(stand_index)), len(sit_index), len(stand_index)))

			# identify the index at which data is to be split between train and test sets
			split_anchor_idx = round(len(sit_index) * 0.8)  # 0.8 = 80% train; 20% test

			train_index = sit_index[:split_anchor_idx] + stand_index[:split_anchor_idx]
			test_index = sit_index[split_anchor_idx:] + stand_index[split_anchor_idx:]
			random.shuffle(train_index)
			random.shuffle(test_index)

			# debug
			for index in test_index:
				plot_image(array_list[index], str(image_filename.values[index]), key)  # str(cl.values[index])
			for index in train_index:
				plot_image(array_list[index], str(image_filename.values[index]), key)

			# convert train & test data sets to 1D arrays and write them to file
			train_set = []
			for index in train_index:
				# position 0 is class; remainder is the flattened 3D array
				train_set.append(np.concatenate([cl.values[index], np.concatenate(array_list[index]).ravel()]))
			test_set = []
			for index in test_index:
				test_set.append(np.concatenate([cl.values[index], np.concatenate(array_list[index]).ravel()]))
			# write the arrays to file
			csv_out_train = os.path.join(output_dir, key + "-TRAIN.csv")
			csv_out_test = os.path.join(output_dir, key + "-TEST.csv")
			pd.DataFrame(train_set).to_csv(csv_out_train, index=False)
			pd.DataFrame(test_set).to_csv(csv_out_test, index=False)

			print("[{}] data set size: {}\n"
				  "\tTrain set:\t{}\n\tTest set:\t{}\n".
				  format(key, len(train_set) + len(test_set), len(train_set), len(test_set)))

	print("DONE. All key point data sets written to file.\n")