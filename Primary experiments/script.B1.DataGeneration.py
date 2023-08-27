from __future__ import print_function

import os
import gc
import sys
import cv2
import math
import colorsys
import configparser
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot


def display_RGB_image(win_title, rgb_image):
	"""
	Display an RGB image using OpenCV, converting it to BGR channel order.

	Args:
		win_title (str): 			The title for the image display window.
		rgb_image (numpy.ndarray): 	The RGB image represented as a NumPy array.
	"""
	cv2.imshow(win_title, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
	cv2.waitKey(1)


def display_DEPTH_image(win_title, depth_image):
	"""
	This function takes a depth image and displays it using the OpenCV library.

	Args:
		win_title (str): 				The title for the image display window.
		depth_image (numpy.ndarray): 	The depth image represented as a NumPy array.
	"""
	cv2.imshow(win_title, depth_image)
	cv2.waitKey(1)


def create_directories(paths):
	"""
	This function checks each path in the input list and creates the directories
	if they are not already present. It ensures that the specified paths are available
	for storing files or other data.

	Args:
		paths (list):   List of directory paths to be created.
	"""
	for path in paths:
		if not os.path.exists(path):
			os.makedirs(path)


def filter_greyscale(image):
	"""
	This function takes an image in the BGR color space (used by OpenCV) and converts
	it to greyscale using the cv2.COLOR_BGR2GREY conversion.

	Args:
		image (numpy.ndarray): 	The input image represented as a NumPy array.

	Returns:
		numpy.ndarray: The greyscale version of the input image.
	"""
	return cv2.cvtColor(image, cv2.COLOR_BGR2GREY)


def filter_gaussian(image):
	"""
	Apply Gaussian blur to an image for noise reduction and smoothing
	that reduces noise and image details.

	Args:
		image (numpy.ndarray): The input image represented as a NumPy array.

	Returns:
		numpy.ndarray: The image after applying Gaussian blur.
	"""
	return cv2.GaussianBlur(image, (11, 11), 0)


def filter_median(image):
	"""
	Applies a median blur to the input image, which replaces each pixel's value
	with the median value of its neighboring pixels. This helps in reducing noise and mitigating
	the salt and pepper effect in the image.

	Args:
		image (numpy.ndarray): The input image represented as a NumPy array.

	Returns:
		numpy.ndarray: The image after applying median blur.
	"""
	return cv2.medianBlur(image, 3)  # specifically for depth images


def fixed_BG_SUB(im, bg):
	"""
	Perform fixed background subtraction to separate foreground objects from a static background in an image.

	Args:
		im (numpy.ndarray): 	The input image represented as a NumPy array.
		bg (numpy.ndarray): 	The background image represented as a NumPy array.

	Returns:
		image tuple: A tuple containing the binary mask and the extracted foreground image.
	"""
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
	"""
	Perform adaptive background subtraction to extract moving objects from a changing background.

	Args:
		im (numpy.ndarray): 	The input image represented as a NumPy array.

	Returns:
		image tuple: A tuple containing the binary mask and the extracted foreground image.
	"""
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
	"""
	Stack two images along the depth (channel) dimension to create a composite image.

	Args:
		im_rgb (numpy.ndarray): 	The RGB image represented as a NumPy array.
		im_depth (numpy.ndarray): 	The depth image represented as a NumPy array.

	Returns:
		numpy.ndarray: The composite image resulting from stacking the input images.
	"""
	return np.dstack((im_rgb, im_depth))


def unstack_layers(im_4d):
	"""
	Separate RGB and depth channels from a stacked 4D array.

	Args:
		im_4d (numpy.ndarray): The stacked 4D array representing the composite image.

	Returns:
		numpy.ndarray: A 3D array containing the separated RGB channels and depth channel.
	"""
	return np.dstack((im_4d[:, :, 0:2], im_4d[:, :, 3]))


def stack_toString(im_class, im_stack):
	"""
	Flatten an image array and concatenate the class label to create a comma-separated string.

	Args:
		im_class (int):				The class label.
		im_stack (numpy.ndarray): 	The stacked 3D/4D image array represented as a NumPy array.

	Returns:
		str: A comma-separated string containing the class label and flattened image values.
	"""
	arr_out = np.concatenate([im_class, im_stack.reshape(-1)])  # append class to flattened image array
	return ','.join('%d' % x for x in arr_out)


def print_stats(df_keypoints, stats_file):
	"""
	This function takes a DataFrame containing data samples with class labels and a statistics file,
	and reports the number of data samples by class both to the console and the specified log file.

	Args:
		df_keypoints (pandas.DataFrame): 	The DataFrame containing data samples and class labels.
		stats_file (file): 					The log file where the statistics will be written.
	"""
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
	hlen = len(hex)
	return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def get_spaced_rgb_colours(N):
	"""
	Generate a list of spaced RGB colours obtained from the HSV colour space,
	and then converting them into RGB tuples.

	Args:
		N (int): 	The number of spaced RGB colours to generate.

	Returns:
		list: A list of RGB tuples representing spaced colours.
	"""
	spacedRGB = []
	spacedHex = get_spaced_hex_colours(N)
	for hex in spacedHex:
		rgb = hex_to_rgb(hex)
		spacedRGB.append(rgb)
	return spacedRGB


def mix_two_rgb(r1, g1, b1, r2, g2, b2):
	"""
	Calculate the blended RGB colour by taking an equal weighted average of the corresponding
	red, green, and blue components of two input RGB colours. If the calculated value exceeds 255.
	The components are restricted to 255 to ensure valid RGB colour values.

	Args:
		r1 (int): The red component of the first RGB colour (0-255).
		g1 (int): The green component of the first RGB colour (0-255).
		b1 (int): The blue component of the first RGB colour (0-255).
		r2 (int): The red component of the second RGB colour (0-255).
		g2 (int): The green component of the second RGB colour (0-255).
		b2 (int): The blue component of the second RGB colour (0-255).

	Returns:
		tuple: An RGB tuple representing the blended colour.
	"""
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
	"""
	Calculates a radial color gradient by mapping the position (x, y) relative to a center
	point (cx, cy) to an HSV hue value. The hue value is then converted to an RGB color.

	Args:
		x (int): 	The x-coordinate of the point for which the radial gradient color is generated.
		y (int): 	The y-coordinate of the point for which the radial gradient color is generated.
		cx (int): 	The x-coordinate of the center point of the radial gradient.
		cy (int): 	The y-coordinate of the center point of the radial gradient.

	Returns:
		tuple: An RGB tuple representing the color gradient at the given point.
	"""
	rx = x - cx
	ry = y - cy
	h = ((math.atan2(ry, rx) / math.pi) + 2.17) / 2.0
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


def get_radial_segment(x, y, cx, cy):
	"""
	Calculates a radial color segment by mapping the position (x, y) relative to a center
	point (cx, cy) to an HSV hue value. The hue value is then converted to an RGB color.

	Args:
		x (int): 	The x-coordinate of the point for which the radial segment color is generated.
		y (int): 	The y-coordinate of the point for which the radial segment color is generated.
		cx (int): 	The x-coordinate of the center point of the radial segment.
		cy (int): 	The y-coordinate of the center point of the radial segment.

	Returns:
		tuple: An RGB tuple representing the color segment at the given point.
	"""
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
	"""
	Calculates a ringed circular colour gradient by mapping the distance between the position (x, y)
	and	the center point (cx, cy) to an HSV hue value. The hue value is then converted to an RGB colour.

	Args:
		x (int): 	The x-coordinate of the point for which the ring gradient color is generated.
		y (int): 	The y-coordinate of the point for which the ring gradient color is generated.
		cx (int): 	The x-coordinate of the center point of the ring gradient.
		cy (int): 	The y-coordinate of the center point of the ring gradient.

	Returns:
		tuple: An RGB tuple representing the colour of the circular gradient at the given point.
	"""
	rx = x - cx
	ry = y - cy
	h = np.sqrt(rx ** 2.0 + ry ** 2.0)
	h = h / (np.sqrt(2) * min(x_axis, y_axis) / 2)

	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


def get_ring_segment(x, y, cx, cy):
	"""
	Calculates a ringed and segmented circular color gradient by mapping the distance between the position (x, y)
	and	the center point (cx, cy) to an HSV hue value. The hue value is then converted to an RGB colour.

	Args:
		x (int): 	The x-coordinate of the point for which the ring segment color is generated.
		y (int): 	The y-coordinate of the point for which the ring segment color is generated.
		cx (int): 	The x-coordinate of the center point of the ring segment.
		cy (int): 	The y-coordinate of the center point of the ring segment.

	Returns:
		tuple: An RGB tuple representing the color of the circular segment at the given point.
	"""
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


def generate_colour_plot(plot_type, centroid):
	"""
	Generates a color plot based on the specified plot type and the centroid position. The color plot
	is created by mapping colors to each pixel in the image based on its location relative to the centroid.

	Args:
		plot_type (str): 	The type of color plot to generate.
							Valid options are 'radial_gradient', 'radial_segment', 'ring_gradient', and 'ring_segment'
		centroid (tuple): 	A tuple containing the x and y coordinates of the centroid point.

	Returns:
		numpy.ndarray: A NumPy array representing the generated color plot image.
	"""
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
	"""
	Generates an image of each type of augmentation colour plot to illustrate their layout.

	Args:
		plot_type (str): 	The type of color plot to generate.
							Valid options are 'radial_gradient', 'radial_segment', 'ring_gradient', and 'ring_segment'
		centroid (tuple): A tuple containing the x and y coordinates of the centroid point.

	Returns:
		numpy.ndarray: A NumPy array representing the generated color plot image.
	"""
	for plot_type in data_types[2:]:
		cx, cy = x_axis / 2, y_axis / 2
		bg_im = generate_color_plot(plot_type, (cx, cy))
		Image.fromarray(bg_im).save(f"{dir}/{plot_type}.png")


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
	arr_out = [(x, y)]

	# create cross shape
	for stride in range(1, d+1):

		# right
		if 0 <= x + stride < x_axis:
			arr_out.append((x + stride, y))
			# create a thicker line
			# for i in range(math.trunc(d/3)+1):
			# 	arr_out.append((x + stride, y + i))
			# 	arr_out.append((x + stride, y - i))

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
	"""
	This function displays the input image along with the specified title.
	The axis is turned off to provide a clean image display.

	Args:
		title (str): The title to be displayed above the image.
		img (numpy.ndarray): A 3D array representing the RGB pixel values of the image.
	"""
	pyplot.imshow(img)
	pyplot.title(title)
	pyplot.axis('off')
	pyplot.show()


def get_coordinates_shape(x, y, d=2):
	""" Specify a chosen coordinate shape that is to be used throughout the script """
	# return get_dot_coordinates(x, y)
	# return get_diamond_coordinates(x, y, d)
	return get_crosshair_coordinates(x, y, d)


def plot_rgb_coordinates(image, coordinates, colour):
	"""
	This function takes an input image, a list of (x, y) coordinates, and an RGB color value.
	It modifies the image by either setting the RGB color at the specified coordinates to the provided
	color or blending it with the existing color if a color is already present at those coordinates.

	Args:
		image (numpy.ndarray): 	A 3D array representing the RGB pixel values of the image.
		coordinates (list): 	A list of (x, y) coordinates to modify.
		colour (tuple): 		An RGB color value (r, g, b) to set or blend.

	Returns:
		numpy.ndarray: The modified image.
	"""
	r, g, b = colour
	for x, y in coordinates:
		r0, g0, b0 = image[y, x]
		if r0 == g0 == b0 == BG_colour:
			image[y, x] = (r, g, b)  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = mix_two_rgb(r, g, b, r0, g0, b0)
	return image


def plot_monochrome_coordinates(image, coordinates, mono_shade):
	"""
	This function takes an input image, a list of (x, y) coordinates, and a monochrome shade value.
	It modifies the image by either setting the shade at the specified coordinates to the provided
	monochrome shade or blending it with the existing shade if a shade is already present at those coordinates.

	Args:
		image (numpy.ndarray): 	A 2D array representing the monochrome pixel values of the image.
		coordinates (list): 	A list of (x, y) coordinates to modify.
		mono_shade (int): 		A monochrome shade value to set or blend.

	Returns:
		numpy.ndarray: The modified image.
	"""
	for x, y in coordinates:
		m0 = image[y, x]
		if m0 == BG_colour:
			image[y, x] = mono_shade  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = ((m0 + mono_shade) / 2)
	return image


def overlay_mask_on_image(fg, mask):
	"""
	This function takes a foreground image and a mask of key points (possibly with color channels)
	and overlays the mask onto the foreground image at corresponding pixel coordinates. It modifies
	the foreground image by replacing pixels with key points from the mask.
	
	Args:
		foreground (numpy.ndarray): 	A 3D array representing the foreground image (RGB or greyscale).
		mask (numpy.ndarray): 			A 3D or 2D array representing the mask of key points (RGB or greyscale).
	
	Returns:
		numpy.ndarray: The modified foreground image with the mask overlay.
	"""
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
	"""
	This function calculates an appropriate crop region based on the centroid position of the silhouette
	in the image using the rule of thirds.

	Args:
		image (numpy.ndarray): 	The input image to be cropped (RGB or greyscale).
		centroid (tuple): 		A tuple containing the x and y coordinates of the body centroid.

	Returns:
		numpy.ndarray: The cropped image.
	"""
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

	if image.ndim == 3:  # RGB image
		image = image[int(y_start):int(y_end), int(x_start):int(x_end), :]
	else:  # Depth image
		image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
	return image


def generate_representation(df, centroid_coor, fg_rgb, fg_depth, datatype):
	"""
	Generates data samples through background subtraction and key point plotting

	Args:
		df (pandas.DataFrame): 		DataFrame containing key point coordinates and confidences.
		centroid_coor (tuple): 		Coordinates of the centroid.
		fg_rgb (numpy.ndarray): 	Foreground RGB image obtained through background subtraction.
		fg_depth (numpy.ndarray):	Foreground depth image obtained through background subtraction.
		datatype (int): 			Chosen augmentation type.

	Returns:
		Tuple of NumPy arrays (numpy.ndarray)
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
	"""
	Identify the body centroid as the midpoint between the neck (1), R-hip (8), and L-hip (11) coordinates.

	Args:
		cor_x (pandas.DataFrame): 		DataFrame containing x-coordinates of key points.
		cor_y (pandas.DataFrame): 		DataFrame containing y-coordinates of key points.
		cor_conf (pandas.DataFrame): 	DataFrame containing confidence levels of key points.
		class_df (pandas.DataFrame): 	DataFrame containing class information.

	Returns:
		Tuple of integers: Coordinates of the identified body centroid.
	"""
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

		# mid-hip is not detected
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
	# read arguments from config file
	config = configparser.ConfigParser()
	config.read('parameters.conf')

	# names of each dataset (0_Baseline, 1_JointColour, 2_RadialGradient, etc.)
	data_types = list(config.get('Global', 'data_types').split(', '))

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
		stats_file.write("DATA PARTITION: \t" + data_partition + "\n")
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
