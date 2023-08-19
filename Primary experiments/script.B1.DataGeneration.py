from __future__ import print_function
from matplotlib import pyplot
from PIL import Image
import pandas as pd
import numpy as np
import configparser
import colorsys
import math
import cv2
import sys
import gc
import os


def display_RGB_image(win_title, rgb_image):
	# cv2 handles images in BGR channel order - therefore, to display RGB correctly, the image must be converted
	cv2.imshow(win_title, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
	cv2.waitKey(1)


def display_DEPTH_image(win_title, depth_image):
	# cv2 handles images in BGR channel order - therefore, to display RGB correctly, the image must be converted
	cv2.imshow(win_title, depth_image)
	cv2.waitKey(1)


def create_directories(paths):
	""" Create directories in the path that don't exist """
	for path in paths:
		if not os.path.exists(path):
			os.makedirs(path)


def filter_greyscale(image):
	""" Convert an image from the BGR color space (used by OpenCV) to greyscale """
	return cv2.cvtColor(image, cv2.COLOR_BGR2GREY)


def filter_gaussian(image):
	""" Applies Gaussian blur to smooth image and reduce noise & image detail """
	return cv2.GaussianBlur(image, (11, 11), 0)


def filter_median(image):
	""" Applies a median blur to smooth image by replacing each pixel's value with the
	median value of its neighbouring pixels and reduces salt & pepper effect """
	return cv2.medianBlur(image, 3)  # specifically for depth images


def fixed_BG_SUB(im, bg):
	""" Performs fixed background subtraction to separate foreground objects from a static background in an image """
	if len(im.shape) == 3:  # RGB IMAGE
		# absolute difference between the the video frame and its background image is computed
		im_sub = cv2.absdiff(filter_gaussian(filter_greyscale(im)), filter_gaussian(filter_greyscale(bg)))
		# threshold is applied to the mean of the difference image, resulting in a binary mask
		im_mask = cv2.threshold(im_sub, np.mean(im_sub), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		# binary mask is dilated to connect regions and fill in gaps using
		im_mask = cv2.dilate(im_mask, None, iterations=3)
		# convert the binary mask to a 3-channel image (BGR)
		im_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)
		# foreground is extracted from the input image based on identified difference region
		im_fg = cv2.cvtColor(cv2.bitwise_and(im_rgb, im_mask), cv2.COLOR_BGR2RGB)
		if verbose:
			display_RGB_image("Fixed BGSub: RGB Mask", im_mask)
			display_RGB_image("Fixed BGSub: RGB Foreground", im_fg)
		return im_mask, im_fg
	else:  # GREYSCALE (DEPTH) IMAGE
		im_sub = cv2.absdiff(im, bg)
		im_mask = cv2.threshold(im_sub, np.mean(im_sub), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		im_mask = filter_median(im_mask)
		im_fg = cv2.bitwise_and(im_depth, im_mask)
		if verbose:
			display_DEPTH_image("Fixed BGSub: DEPTH Mask", im_mask)
			display_DEPTH_image("Fixed BGSub: DEPTH Foreground", im_fg)
		return im_mask, im_fg


def adaptive_BG_SUB(im):
	""" Adaptive background subtraction extracts moving objects from a varying or changing background """
	learning_rate = 0.004  # float(set_num)/frame_history
	if len(im.shape) == 3:  # RGB image
		# apply adaptive background subtraction using OpenCV class
		im_mask = subtractor_rgb.apply(im, learningRate=learning_rate)
		# remove shadows by converting grey shadow pixels (marked as 127) to black (0)
		im_mask[im_mask == 127] = 0
		# convert the binary mask to a 3-channel image (BGR) to be substracted from the
		im_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)
		# foreground is extracted from the input image based on identified difference region
		im_fg = cv2.cvtColor(cv2.bitwise_and(im_rgb, im_mask), cv2.COLOR_BGR2RGB)
		if verbose:
			display_RGB_image("Adapt BGSub: RGB Mask", im_mask)
			display_RGB_image("Adapt BGSub: RGB Foreground", im_fg)
		return im_mask, im_fg
	else:  # GREYSCALE (DEPTH) IMAGE
		im_mask = subtractor_depth.apply(im_depth, learningRate=learning_rate)
		im_fg = cv2.bitwise_and(im_depth, im_mask)
		if verbose:
			display_DEPTH_image("Fixed BGSub: DEPTH Mask", im_mask)
			display_DEPTH_image("Fixed BGSub: DEPTH Foreground", im_fg)
		return im_mask, im_fg


def stack_layers(im_rgb, im_depth):
	""" Stacks two images together along the depth (channel) dimension to create a new composite image (4D array) """
	return np.dstack((im_rgb, im_depth))


def unstack_layers(im_4d):
	""" Separates the first three channels (RGB) and fourth cahneel (depth) from a stacked 4D array """
	return np.dstack((im_4d[:, :, 0:2], im_4d[:, :, 3]))


def stack_toString(im_class, im_stack):
	""" flatten the image array and concatenate the class label to a comma-separated string of values """
	arr_out = np.concatenate([im_class, im_stack.reshape(-1)])  # append class to flattened image array
	return ','.join('%d' % x for x in arr_out)


def print_stats(df_keypoints, stats_file):
	""" Report the number of data samples by class both to command-line and log file """
	class_mapping = {
		1: 'Stand',
		2: 'Sit',
		3: 'Lying',
		4: 'Bending',
		5: 'Crawling',
		6: 'Empty'
	}

	# initialise a dictionary (class_indexes) with empty lists for each class number
	class_indexes = {cls: [] for cls in class_mapping.keys()}

	# for each row in df, append its index to the corresponding list based on value of ' class (number)'
	for index, cls in enumerate(df_keypoints[' class (number)']):
		if cls in class_mapping:
			class_indexes[cls].append(index)
		else:
			print("Unknown class: " + str(cls))

	total_sample_size = sum(len(indexes) for indexes in class_indexes.values())

	# print class counts to console
	print("\t{:<20}: \t{}".format("TOTAL SAMPLE SIZE", total_sample_size))
	for cls, indexes in class_indexes.items():
		print("\t  {:<10}\t\t: \t{}".format(class_mapping[cls], len(indexes)))
	print("\n")

	# write class counts to file
	stats_file.write("\t{:<20}: \t{}\n".format("TOTAL SAMPLE SIZE", total_sample_size))
	for cls, indexes in class_indexes.items():
		stats_file.write("\t  {:<10}\t\t: \t{}\n".format(class_mapping[cls], len(indexes)))
	stats_file.write("\n")


def get_spaced_hex_colours(N):
	""" Generate visually distinct colours that are evenly distributed in the HSV colour space """
	HSV_tuples = [(x * 1.0 / N, 1.0, 1.0) for x in range(N)]
	hex_out = []
	for rgb in HSV_tuples:
		rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
		hex_out.append('#%02x%02x%02x' % tuple(rgb))
	return hex_out


def hex_to_rgb(hex):
	""" Convert a hexadecimal colour code into an RGB tuple """
	hex = hex.lstrip('#')
	hlen = len(hex)
	return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def get_spaced_rgb_colours(N):
	""" Generates a list of spaced RGB colours obtained from the HSV colour space """
	spacedRGB = []
	spacedHex = get_spaced_hex_colours(N)
	for hex in spacedHex:
		rgb = hex_to_rgb(hex)
		spacedRGB.append(rgb)
	return spacedRGB


# rgb operates on additive colour channels
def mix_two_rgb(r1, g1, b1, r2, g2, b2):
	""" Blends RGB colours by taking an equal weighted average of their respective R, G, and B components """
	tmp_r = (r1 * 0.5) + (r2 * 0.5)
	tmp_g = (g1 * 0.5) + (g2 * 0.5)
	tmp_b = (b1 * 0.5) + (b2 * 0.5)

	# if calculated value is greater than 255, set it back to its original input values
	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def get_radial_gradient(x, y, cx, cy):
	"""  Generate radial color gradient based on the position (x, y) relative to a center point (cx, cy) """
	rx = x - cx
	ry = y - cy
	h = ((math.atan2(ry, rx) / math.pi) + 2.17) / 2.0
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


# wheel segment
def get_radial_segment(x, y, cx, cy):
	"""  Generate radial colour segments based on the position (x, y) relative to a center point (cx, cy) """
	rx = x - cx
	ry = y - cy
	h = ((math.atan2(ry, rx) / math.pi) - 2.0) / 2.0
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	lst = list(rgb)
	for idx, val in enumerate(lst):
		lst[idx] = int(round(val * 255.0))
		if lst[idx] > 255:
			lst[idx] = 255
		if lst[idx] < 0:
			lst[idx] = 0
	return tuple(lst)


def get_ring_gradient(x, y, cx, cy):
	""" Generate circular colour gradient based on the position (x, y) relative to a center point (cx, cy) """
	rx = x - cx
	ry = y - cy
	h = np.sqrt(rx ** 2.0 + ry ** 2.0)
	h = h / (np.sqrt(2) * min(x_axis, y_axis) / 2)

	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


def get_ring_segment(x, y, cx, cy):
	"""  Generate circular colour segments based on the position (x, y) relative to a center point (cx, cy) """
	radius = min(x_axis, y_axis) / 2
	cx = cx - 0.5
	cy = cy - 0.5

	num_rings = 5  # change this for more rings
	num_colours = num_rings + 1
	increment = round(radius / num_rings) + 0.1
	hue_increment = 360 / num_colours

	# calculate the Euclidean distance s from the point (x, y) to the center (cx, cy)
	rx = x - cx
	ry = y - cy
	s = round(np.sqrt(rx ** 2.0 + ry ** 2.0))

	# determine the quadrant/segment in which s lies. Based on the limits each ring, calculate the hue its contained pixel
	if s <= increment * num_rings:
		for i in range(num_colours):
			if s >= increment * i and s < increment * (i + 1):
				h = hue_increment * i
	else:
		h = hue_increment * num_rings

	h = h / 360  # normalize the hue value to the range [0, 1] by dividing by 360
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


def generate_color_plot(plot_type, centroid):
	""" Generates an image with the specified type of colour plot and returns the resulting image as a NumPy array """
	im = np.full((y_axis, x_axis, 3), BG_colour)
	cx, cy = centroid

	plot_type_indices = {
		data_types[2]: get_radial_gradient,
		data_types[3]: get_radial_segment,
		data_types[4]: get_ring_gradient,
		data_types[5]: get_ring_segment
	}
	if plot_type not in plot_type_indices:
		raise ValueError("Invalid plot type")
	# assign plot function to variable
	color_func = plot_type_indices[plot_type]

	# loop through each image pixel and calculate its colour based on its location
	for x in range(im.shape[1]):
		for y in range(im.shape[0]):
			im[y, x] = color_func(x, y, cx, cy)

	# ensure that the pixel values in the array are within the valid range [0, 255]
	im = np.array(im, dtype=np.uint8)
	return im


def generate_augmentation_illustrations(dir):
	""" Generate each of the colour wheel augmentations to illustrate their layout """
	for plot_type in data_types[2:]:
		cx, cy = x_axis / 2, y_axis / 2
		bg_im = generate_color_plot(plot_type, (cx, cy))
		Image.fromarray(bg_im).save(f"{dir}/{plot_type}.png")


def get_dot_coordinates(x, y):
	""" Returns the x, y coordinate as a tuple """
	arr_out = [(x, y)]
	return arr_out


def get_diamond_coordinates(x, y, d):
	""" Generates a set of coordinates forming a diamond shape around a central point """
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

	arr_out = list(set(arr_out))
	return arr_out


def get_crosshair_coordinates(x, y, d):
	""" Generates a set of coordinates forming a crosshair shape around a central point """
	arr_out = [(x, y)]

	# create cross shape
	for stride in range(1, d+1):

		# right
		if 0 <= x + stride < x_axis:
			arr_out.append((x + stride, y))

		# left
		if 0 <= x - stride < x_axis:
			arr_out.append((x - stride, y))

		# up
		if 0 <= y + stride < y_axis:
			arr_out.append((x, y + stride))

		# down
		if 0 <= y - stride < y_axis:
			arr_out.append((x, y - stride))

	arr_out = list(set(arr_out))
	return arr_out


def plot_image(title, img):
	""" Simple plot image function that takes a 3D array of RGB pixel values """
	pyplot.imshow(img)
	pyplot.title(title)
	pyplot.axis('off')
	pyplot.show()


def get_coordinates_shape(x, y, d=2):
	""" Elect a coordinate shape that is to be used throughout the script """
	# return get_dot_coordinates(x, y)
	# return get_diamond_coordinates(x, y, d)
	return get_crosshair_coordinates(x, y, d)


def plot_rgb_coordinates(image, coordinates, colour):
	""" Modify image by changing rgb colour values at specific coordinates """
	r, g, b = colour
	for x, y in coordinates:
		r0, g0, b0 = image[y, x]
		if r0 == g0 == b0 == BG_colour:
			image[y, x] = (r, g, b)  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = mix_two_rgb(r, g, b, r0, g0, b0)
	return image


def plot_monochrome_coordinates(image, coordinates, mono_shade):
	""" Modify image by changing the shade (brightness value) of specified coordinates """
	for x, y in coordinates:
		m0 = image[y, x]
		if m0 == BG_colour:
			image[y, x] = mono_shade  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = ((m0 + mono_shade) / 2)
	return image


def overlay_mask_on_image(fg, mask):
	""" Overlays a mask of key points onto a foreground image at corresponding pixel coordinates """
	for x in range(mask.shape[1]):
		for y in range(mask.shape[0]):
			if mask.ndim == 3:
				if not mask[y, x][0] == mask[y, x][1] == mask[y, x][2] == BG_colour:
					fg[y, x] = mask[y, x]
			else:
				if not mask[y, x] == BG_colour:
					fg[y, x] = mask[y, x]
	return fg


def crop_image(image, centroid):
	""" Determine an appropriate crop region (by rule of thirds) based on the centroid position of the human silhouette """
	x_centroid, y_centroid = centroid
	x_diff = x_axis-x_crop
	y_diff = y_axis-y_crop
	x_ratio = x_axis/3
	y_ratio = y_axis/3

	x_start = 0
	x_end = 0
	y_start = 0
	y_end = 0

	for i in range(3):
		# CROP X AXIS
		if x_centroid >= x_ratio * i and x_centroid < x_ratio * (i + 1):
			if i == 0:
				x_start = 0
				x_end = x_crop
			elif i == 1:
				x_start = x_diff/2
				x_end = x_axis-x_diff/2
			else:
				x_start = x_diff
				x_end = x_axis
		# CROP Y AXIS
		if y_centroid >= y_ratio * i and y_centroid < y_ratio * (i + 1):
			if i == 0:
				y_start = 0
				y_end = y_crop
			elif i == 1:
				y_start = y_diff / 2
				y_end = y_axis - y_diff / 2
			else:
				y_start = y_diff
				y_end = y_axis

	if image.ndim == 3: # RGB image
		image = image[int(y_start):int(y_end), int(x_start):int(x_end), :]
	else: # Depth image
		image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
	return image


def generate_representation(df, centroid_coor, fg_rgb, fg_depth, datatype):
	""" Generates data samples through background subtraction and key point plotting

	Returns:
		foreground RGB image	: BG subtracted result of original RGB video frame
		foreground depth image	: BG subtracted result of original depth video frame
		key point RGB mask		: Key point plot derived from chosen colour-based augmentation
		key point depth mask	: Key point plot of white pixel values
		cutout image			: Key point plotting against the colour wheel-based augmentation for illustrative purposes
	"""

	if datatype == 0:  # baseline (requires no augmentation)
		return fg_rgb, fg_depth, fg_rgb, fg_depth, None

	# extract necessary values from dataframe
	cor_x = df.filter(regex="x\d+").astype(int)
	cor_y = df.filter(regex="y\d+").astype(int)
	cor_conf = df.filter(regex="c\d+").round(6)
	cl = df[[' class (number)']]

	tmp_rgb_mask = np.full((y_axis, x_axis, 3), BG_colour, np.uint8)
	tmp_mono_mask = np.full((y_axis, x_axis), BG_colour, np.uint8)
	spacedRGB = get_spaced_rgb_colours(cor_x.shape[1])  # get an evenly distributed array of colours

	# cutout images for colour wheels
	cutout_im = None
	if datatype in list([2, 3, 4, 5]):
		cutout_im = generate_color_plot(data_types[datatype], centroid_coor)

	# for every coordinate in the set (thus every column)
	if cl.iloc[0, 0] != 6:  # images without any person in the frame - no joints to map
		for col in range(0, cor_x.shape[1]):
			x = int(cor_x.iloc[0, col])
			# x = int(x / 160 * x_axis)  # scale the x,y coordinates to chosen dimensions
			y = int(cor_y.iloc[0, col])
			# y = int(y / 120 * y_axis)  # scale the x,y coordinates to chosen dimensions
			confidence = cor_conf.iloc[0, col]

			# if the joint is detected (non-occluded body key point)
			if not (x == 0 and y == 0 and confidence == 0):
				if datatype == 1:  # joint colour
					r, g, b = spacedRGB[col]
				if datatype == 2:
					r, g, b = get_radial_gradient(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 3:
					r, g, b = get_radial_segment(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 4:
					r, g, b = get_ring_gradient(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 5:
					r, g, b = get_ring_segment(x, y, centroid_coor[0], centroid_coor[1])

				# plot joint as a shape represented as a set of coordinates
				xy_coordinates = get_coordinates_shape(x, y)
				tmp_rgb_mask = plot_rgb_coordinates(tmp_rgb_mask, xy_coordinates, [r, g, b])
				tmp_mono_mask = plot_monochrome_coordinates(tmp_mono_mask, xy_coordinates, abs(BG_colour - 255))
				if datatype in list([2, 3, 4, 5]):
					cutout_im = plot_rgb_coordinates(cutout_im, xy_coordinates, [BG_colour, BG_colour, BG_colour])

	# overlay the mask onto the foreground image
	rgb_out = overlay_mask_on_image(fg_rgb, tmp_rgb_mask)
	mono_out = overlay_mask_on_image(fg_depth, tmp_mono_mask)

	return rgb_out, mono_out, tmp_rgb_mask, tmp_mono_mask, cutout_im


def identify_body_centroid(cor_x, cor_y, cor_conf, class_df):
	""" Identify the body centroid as the midpoint between the neck (1), R-hip (8), and L-hip (11) coordinates """
	count_coor = 0
	centroid_x = x_axis / 2  # default if centroid cannot be identified
	centroid_y = y_axis / 2  # default if centroid cannot be identified
	if class_df.iloc[0, 0] != 6:  # if not a empty class representation (no joints), or false/inadequate detection
		# neck is not detected, use closest alternative
		if not cor_conf.iloc[0, 1] > 0.0:  # neck
			if cor_conf.iloc[0, 0] > 0.0:  # nose
				centroid_x = int(cor_x.iloc[0, 0])
				centroid_y = int(cor_y.iloc[0, 0])
				count_coor += 1
			if (cor_conf.iloc[0, 17] > 0.0) or (cor_conf.iloc[0, 18] > 0.0):  # left & right ears
				inner_count = 0
				if cor_conf.iloc[0, 17] > 0.0:
					centroid_x = int(cor_x.iloc[0, 17])
					centroid_y = int(cor_y.iloc[0, 17])
					inner_count += 1
				if cor_conf.iloc[0, 18] > 0.0:
					centroid_x = int(cor_x.iloc[0, 18])
					centroid_y = int(cor_y.iloc[0, 18])
					inner_count += 1
				centroid_x = centroid_x / inner_count
				centroid_y = centroid_y / inner_count
			if (cor_conf.iloc[0, 2] > 0.0) or (cor_conf.iloc[0, 5] > 0.0):  # left & right shoulder
				inner_count = 0
				if cor_conf.iloc[0, 2] > 0.0:
					centroid_x = int(cor_x.iloc[0, 2])
					centroid_y = int(cor_y.iloc[0, 2])
					inner_count += 1
				if cor_conf.iloc[0, 5] > 0.0:
					centroid_x = int(cor_x.iloc[0, 5])
					centroid_y = int(cor_y.iloc[0, 5])
					inner_count += 1
				centroid_x = centroid_x / inner_count
				centroid_y = centroid_y / inner_count
			else:
				print("\t" + im_filename + "\t XXX (missing TORSO joints for centroid)")
		else:  # use neck as upperbody midpoint
			centroid_x = int(cor_x.iloc[0, 1])
			centroid_y = int(cor_y.iloc[0, 1])
			count_coor += 1

		# midhip is not detected
		if not cor_conf.iloc[0, 8] > 0.0:
			if cor_conf.iloc[0, 9] > 0.0:  # RHip
				centroid_x += int(cor_x.iloc[0, 9])
				centroid_y += int(cor_y.iloc[0, 9])
				count_coor += 1
			elif cor_conf.iloc[0, 12] > 0.0:  # LHip
				centroid_x += int(cor_x.iloc[0, 12])
				centroid_y += int(cor_y.iloc[0, 12])
				count_coor += 1
			if (cor_conf.iloc[0, 10] > 0.0) or (cor_conf.iloc[0, 13] > 0.0):  # left & right knees
				inner_count = 0
				if cor_conf.iloc[0, 10] > 0.0:
					centroid_x = int(cor_x.iloc[0, 10])
					centroid_y = int(cor_y.iloc[0, 10])
					inner_count += 1
				if cor_conf.iloc[0, 13] > 0.0:
					centroid_x = int(cor_x.iloc[0, 13])
					centroid_y = int(cor_y.iloc[0, 13])
					inner_count += 1
				centroid_x = centroid_x / inner_count
				centroid_y = centroid_y / inner_count
			else:
				print("\t" + str(df.iloc[0, 0]) + "\t YYY (missing LOWERHALF joints for centroid)")
		else:  # use midhip point as lowerbody midpoint
			centroid_x += int(cor_x.iloc[0, 8])
			centroid_y += int(cor_y.iloc[0, 8])
			count_coor += 1

		# average to obtain midpoint
		if count_coor > 0:
			centroid_x = centroid_x / count_coor
			centroid_y = centroid_y / count_coor
		else:
			# the default midpoint is the half of axis lengths (see variables at start of code block)
			print("\t!!! Reverted to default centroid for image: " + str(df.iloc[0, 0]))

	return int(centroid_x), int(centroid_y)


if __name__ == "__main__":

	data_types = list(["0_Baseline", "1_JointColour", "2_RadialGradient", "3_RadialSegment", "4_RingGradient", "5_RingSegment"])

	# read arguments from config file
	config = configparser.ConfigParser()
	config.read('config.ini')

	# dimensions of input video frame (the image will be resized to this specification)
	x_axis = int(config.get('DataGen', 'x_axis'))
	y_axis = int(config.get('DataGen', 'y_axis'))
	# crop generated output video frame to this dimension
	x_crop = int(config.get('DataGen', 'x_crop'))
	y_crop = int(config.get('DataGen', 'y_crop'))
	# 0: black; 255: white
	BG_colour = int(config.get('DataGen', 'BG_colour'))
	# individual images (video frames)
	video_frames_dir = os.path.abspath(config.get('DataGen', 'video_frames_dir'))
	# CSV files of OpenPose key point predictions
	keypoints_dir = os.path.join(os.path.abspath(config.get('DataGen', 'keypoints_dir')), str(x_axis) + "x" + str(y_axis))
	# output location for generated images
	generated_dir = os.path.join(os.path.abspath(config.get('DataGen', 'generated_dir')), str(x_crop) + "x" + str(y_crop))
	# enable verbsoity to display the intermeidate stages of the data generation process
	verbose = config.getboolean('DataGen', 'verbose')
	# choose to store the RGB and depth frames as separate images which are combined in the RGBA output
	samples_flag = config.getboolean('DataGen', 'samples_flag')
	# generate augmentations and store the image files and CSV files (1D arrays of 4D RGBA image array)
	CSV_flag = config.getboolean('DataGen', 'CSV_flag')
	# generate augementations and store the RGBA image files (combines RGB + depth images)
	RGBA_flag = config.getboolean('DataGen', 'RGBA_flag')

	# generate illustrations of each colour wheel to demonstrate the augmentation scheme
	augmentation_illustration_dir = os.path.join(generated_dir, "augmentation-illustration")
	create_directories([augmentation_illustration_dir])
	generate_augmentation_illustrations(augmentation_illustration_dir)

	# create output directories
	for dataset_id, folder in enumerate(data_types):
		parent_folder = os.path.join(os.path.join(generated_dir, folder), "image-samples")
		if dataset_id == 0:
			create_directories([os.path.join(parent_folder, "RGB"), os.path.join(parent_folder, "DEPTH")])
		else:
			create_directories([os.path.join(parent_folder, "RGB_keypoints-overlayed"), os.path.join(parent_folder, "RGB_keypoints"), os.path.join(parent_folder, "DEPTH_keypoints-overlayed"), os.path.join(parent_folder, "DEPTH_keypoints")])
			if dataset_id in list([2, 3, 4, 5]):
				create_directories([os.path.join(parent_folder, "CUTOUT_keypoints")])
		# create Image dir alongside Samples
		for subfolder in ["empty", "lying", "sitting", "crawling", "bending", "standing"]:
			create_directories([os.path.join(os.path.join(os.path.join(generated_dir, folder), "Validation"), subfolder)])
			create_directories([os.path.join(os.path.join(os.path.join(generated_dir, folder), "Training"), subfolder)])
			create_directories([os.path.join(os.path.join(os.path.join(generated_dir, folder), "Testing"), subfolder)])

	stats_file = open(generated_dir + "/DataGen_DatasetStatistics.txt", "w+")  # stores stats on the number of classes in dataset
	skipframes_file = open(generated_dir + "/DataGen_SkippedFrames.txt", "w+")

	# generate the augmented datasets
	for data_partition in list(['Training', 'Validation', 'Testing']):
		print("\n### Generating data partition:\t" + data_partition + " ###\n")

		# prepare output CSV files (flat array of 3D image pixel arrays)
		csv_files = None
		if CSV_flag:
			csv_files = [open(os.path.join(os.path.join(generated_dir, dataset_name), data_partition + ".csv"), 'w+') for dataset_name in list(data_types)]

		# import key points from CSV files
		keypoint_input_csv = os.path.join(keypoints_dir, "Keypoints_" + data_partition + ".csv")
		df_orig = pd.read_csv(keypoint_input_csv)
		df_orig['Filename'] = df_orig['Filename'].str.replace('_keypoints.json', '')  # remove unwanted part of filename
		stats_file.write("DATA PARTITION: " + data_partition + "\n")
		print_stats(df_orig, stats_file)

		# get the set numbers of the dataset e.g. Training_{1172} : pertains to a specific participant and room
		image_dataset_dir = os.path.join(video_frames_dir, data_partition)
		frame_sets = [os.path.basename(f).split('_')[1] for f in df_orig['Filename'].tolist()]

		# iterate through each set number for perform BG subtraction for that collection of footage
		frame_sets = set(frame_sets)
		skipframes_file.write("\n" + data_partition.upper() + "\n")
		for set_num in frame_sets:
			set_num = str(set_num)

			# FIXED BG SUBTRACTION : DPETH FOOTAGE
			bg_rgb = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "bg"), set_num + "_rgb.png"))
			bg_depth = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "bg"), set_num + "_depth.png"), cv2.IMREAD_GRAYSCALE)

			# ADAPTIVE BG SUBTRACTION : RGB FOOTAGE
			frame_history = 1000
			subtractor_rgb = cv2.createBackgroundSubtractorMOG2(history=frame_history, varThreshold=25, detectShadows=True)
			subtractor_depth = cv2.createBackgroundSubtractorMOG2(history=frame_history, detectShadows=False)

			# identify the image sets (collections of participant and room specific instances)
			filter = data_partition + "_" + set_num
			df_tmp = df_orig[df_orig['Filename'].str.contains(filter)]
			frame_indexes = [os.path.basename(f).split('_')[2] for f in df_tmp['Filename'].tolist()]
			frame_indexes.sort(key=int)

			# iterate through each video frame in set collection
			for idx in frame_indexes:

				# filter dataframe for relevant poses/images (regarding data partition and set number)
				# NOTE: OpenPose predictions were normalised to 160x120 although original video frames are in 320x240
				filter = data_partition + "_" + set_num + "_" + str(idx)  # filters to a single row
				df = df_tmp[df_tmp['Filename'].str.contains(filter)]
				im_filename = df.iloc[0, 0]
				im_filetype = im_filename.split('_')[0]
				im_class = im_filename.split('_')[3]
				if df.shape[0] < 1:
					print("No rows are present in data frame. Row count: " + df.shape[0], file=sys.stderr)
					exit()
				cor_x = df.filter(regex="x\d+").astype(int)
				cor_y = df.filter(regex="y\d+").astype(int)
				cor_conf = df.filter(regex="c\d+").round(6)
				cl = df[[' class (number)']]

				# skip the frame if it has too few valid joints (severe occlusion or false detection)
				if cl.iloc[0, 0] != 6:  # if not a deliberately empty frame
					valid_pts = 0
					for col in range(0, cor_conf.shape[1]):
						if cor_conf.iloc[0, col] > 0.0:
							valid_pts += 1
					if valid_pts <= 13:
						skipframes_file.write(im_filename + "\n")
						continue

				# Body centroid is used in cropping video frames and for generating the augmentations
				# the body centroid is determined as the midpoint between the neck (1), R-hip (8), and L-hip (11) coordinates
				centroid_coor = identify_body_centroid(cor_x, cor_y, cor_conf, cl)

				# GET CURRENT FRAME'S RGB & DEPTH IMAGE
				print(data_partition + "  " + str(set_num) + "  " + str(idx))
				im_rgb = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "rgb"), im_filename))
				im_depth = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "depth"), im_filename), cv2.IMREAD_GRAYSCALE)
				# scale the images from original size to the specified resizing dimensions
				bg_rgb = cv2.resize(bg_rgb, (x_axis, y_axis))
				bg_depth = cv2.resize(bg_depth, (x_axis, y_axis))
				im_rgb = cv2.resize(im_rgb, (x_axis, y_axis))
				im_depth = cv2.resize(im_depth, (x_axis, y_axis))

				# BACKGROUND SUBTRACTION
				# RGB images uses adaptive subtraction; Depth images uses fixed subtraction
				fg_rgb = adaptive_BG_SUB(im_rgb)[1]
				fg_depth = fixed_BG_SUB(im_depth, bg_depth)[1]

				# GENERATE AND SAVE DATA SETS #
				for dataset_id, dataset_name in enumerate(data_types):

					# images returned from generate representation are in RGB order (not BGR)
					im_rgb_out, im_depth_out, im_rgb_mask, im_depth_mask, im_cutout = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_depth_out, centroid_coor))

					if verbose:  # display generated images
						display_RGB_image(data_types[dataset_id] + ": rgb", crop_image(im_rgb_out, centroid_coor))
						display_DEPTH_image(data_types[dataset_id] + ": depth", crop_image(im_depth_out, centroid_coor))

					if CSV_flag:  # write to CSV
						out_string = stack_toString(cl.values[0], out_im)  # add class and flatten
						csv_files[dataset_id].write(out_string + "\n")

					if RGBA_flag:  # save RGBA image
						im_path = os.path.join(os.path.join(generated_dir, data_types[dataset_id]), im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)

					if samples_flag:  # store sample images (saves RGB and depth images as separate files)
						out_path = os.path.join(generated_dir, data_types[dataset_id], "image-samples")
						if dataset_id == 0:
							# images are in RGB, but cv2 handles images in BGR channel order (therefore, images must be converted to BRG to store correctly)
							cv2.imwrite(os.path.join(out_path, "RGB", im_filename), crop_image(cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR), centroid_coor))
							cv2.imwrite(os.path.join(out_path, "DEPTH", im_filename), crop_image(fg_depth, centroid_coor))
						else:
							# images are in RGB, but cv2 handles images in BGR channel order (therefore, images must be converted to BRG to store correctly)
							cv2.imwrite(os.path.join(out_path, "RGB_keypoints", im_filename), crop_image(cv2.cvtColor(im_rgb_mask, cv2.COLOR_RGB2BGR), centroid_coor))
							cv2.imwrite(os.path.join(out_path, "RGB_keypoints-overlayed", im_filename), crop_image(cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR), centroid_coor))
							cv2.imwrite(os.path.join(out_path, "DEPTH_keypoints", im_filename), crop_image(im_depth_mask, centroid_coor))
							cv2.imwrite(os.path.join(out_path, "DEPTH_keypoints-overlayed", im_filename), crop_image(im_depth_out, centroid_coor))
							if dataset_id in list([2, 3, 4, 5]):
								Image.fromarray(crop_image(np.array(im_cutout, dtype=np.uint8), centroid_coor)).save(os.path.join(out_path, "CUTOUT_keypoints", im_filename))

				# encourage garbage collection to keep RAM from becoming full
				gc.collect()
			gc.collect()
