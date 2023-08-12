from __future__ import print_function
import os
import pandas as pd
from PIL import Image
import numpy as np
import colorsys
import math
import scipy.ndimage
from matplotlib import pyplot
import sys
import random
import gc
import cv2
import glob

# import custom module from lib directory
from shared_utils import *
sys.path.insert(0, os.path.abspath('../lib'))

# Description:	V2.1.0 Stores the generated images into directories that can be provided as paths for training,
#				testing, validation in Keras while also


# Create output directories - easy to delete the folder and start over
def create_directories(paths):
	for path in paths:
		if not os.path.exists(path):
			os.makedirs(path)


def filter_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def filter_bilateral(image):
	return cv2.bilateralFilter(image, 7, 150, 150)


def filter_gaussian(image):
	return cv2.GaussianBlur(image, (11, 11), 0)


# should be done on mask to filter out salt and pepper effect
def filter_median(image):
	return cv2.medianBlur(image, 3)


# dilate then erode image
def filter_close_operation(image):
	kernel = np.ones((5, 5), np.uint8)
	return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def fixed_BG_SUB(im, bg):
	if len(im.shape) == 3:  # colour image
		# RGB #
		# find difference on two grayscale images that are slightly smoothed using filter and threshold on the mean of difference image
		im_sub = cv2.absdiff(filter_gaussian(filter_grayscale(im)), filter_gaussian(filter_grayscale(bg)))
		im_mask = cv2.threshold(im_sub, np.mean(im_sub), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		im_mask = cv2.dilate(im_mask, None, iterations=3)
		im_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)  # convert mask to 3 channels
		im_fg = cv2.bitwise_and(im_rgb, im_mask)  # extract the white region
		return im_mask, im_fg
	else:
		# DEPTH #
		im_sub = cv2.absdiff(im, bg)
		im_mask = cv2.threshold(im_sub, np.mean(im_sub), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		im_mask = filter_median(im_mask)
		im_fg = cv2.bitwise_and(im_depth, im_mask)  # extract the white region
		return im_mask, im_fg


def adaptive_BG_SUB(im):
	learning_rate = 0.004 # float(set_num)/frame_history
	if len(im.shape) == 3:  # colour image
		# RGB #
		im_mask = subtractor_rgb.apply(im, learningRate=learning_rate)
		im_mask[im_mask == 127] = 0  # remove shadows, marked with value 127, converted to black
		im_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)  # convert mask to 3 channels
		im_fg = cv2.bitwise_and(im_rgb, im_mask)  # extract the white region against original image
		return im_mask, im_fg
	else:
		# DEPTH #
		im_mask = subtractor_depth.apply(im_depth, learningRate=learning_rate)
		im_fg = cv2.bitwise_and(im_depth, im_mask)  # extract the white region
		return im_mask, im_fg


# stack the rgb and depth image into a 4d array
def stack_layers(im_rgb, im_depth):
	return np.dstack((im_rgb, im_depth))


# stack the rgb and depth image into a 4d array
def unstack_layers(im_4d):
	return np.dstack((im_4d[:, :, 0:2], im_4d[:, :, 3]))


# convert 4d stacked image layers to a 1d array
def stack_toString(im_class, im_stack):
	arr_out = np.concatenate([im_class, im_stack.reshape(-1)])  # append class to 1d image array
	return ','.join('%d' % x for x in arr_out)


# take 1d toString array and seperate class and rgb and depth layers
def string_toNumpy(arr_1d):
	# read array back in
	im_class = arr_1d[0]  # get class as first element
	arr_in = np.delete(arr_1d, 0)  # remove prepended class
	arr_in = np.reshape(arr_in, (240, 320, 4))  # reshape to 4 dimensional array
	im_rgb, im_d = np.array(arr_in[:, :, 0:3], dtype=np.uint8), np.array(arr_in[:, :, 3], dtype=np.uint8)
	return im_class, im_rgb, im_depth


def print_stats(df_keypoints, stats):
	# class of each image
	cl = df_keypoints[[' class (number)']]
	stand_indexes = []  # 1
	sit_indexes = []  # 2
	lying_indexes = []  # 3
	bending_indexes = []  # 4
	crawling_indexes = []  # 5
	empty_indexes = []  # 6

	for index, cls in enumerate(cl.values):
		if cls == 1:
			stand_indexes.append(index)
		elif cls == 2:
			sit_indexes.append(index)
		elif cls == 3:
			lying_indexes.append(index)
		elif cls == 4:
			bending_indexes.append(index)
		elif cls == 5:
			crawling_indexes.append(index)
		else:
			empty_indexes.append(index)

	# report statistics
	print("\tTOTAL SAMPLE SIZE: \t{}".format(
		len(stand_indexes) + len(sit_indexes) + len(lying_indexes) + len(bending_indexes) + len(crawling_indexes) + len(
			empty_indexes)))
	print("\t  Stand   \t\t: \t{}".format(len(stand_indexes)))
	print("\t  Sit     \t\t: \t{}".format(len(sit_indexes)))
	print("\t  Lying   \t\t: \t{}".format(len(lying_indexes)))
	print("\t  Bending \t\t: \t{}".format(len(bending_indexes)))
	print("\t  Crawling\t\t: \t{}".format(len(crawling_indexes)))
	print("\t  Empty   \t\t: \t{}\n".format(len(empty_indexes)))

	stats.write("\tTOTAL SAMPLE SIZE: \t\t\t{}\n".format(
		len(stand_indexes) + len(sit_indexes) + len(lying_indexes) + len(bending_indexes) + len(crawling_indexes) + len(
			empty_indexes)))
	stats.write("\t  Stand sample size: \t\t{}\n".format(len(stand_indexes)))
	stats.write("\t  Sit sample size: \t\t\t{}\n".format(len(sit_indexes)))
	stats.write("\t  Lying and sample size: \t{}\n".format(len(lying_indexes)))
	stats.write("\t  Bending sample size: \t\t{}\n".format(len(bending_indexes)))
	stats.write("\t  Crawling sample size: \t{}\n".format(len(crawling_indexes)))
	stats.write("\t  Empty sample size: \t\t{}\n\n".format(len(empty_indexes)))


def get_spaced_hex_colours(N):
	HSV_tuples = [(x * 1.0 / N, 1.0, 1.0) for x in range(N)]
	hex_out = []
	for rgb in HSV_tuples:
		rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
		hex_out.append('#%02x%02x%02x' % tuple(rgb))
	return hex_out


def hex_to_rgb(hex):
	hex = hex.lstrip('#')
	hlen = len(hex)
	return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def get_spaced_rgb_colours(N):
	# create RGB colours
	spacedRGB = []
	spacedHex = get_spaced_hex_colours(N)
	for hex in spacedHex:
		rgb = hex_to_rgb(hex)
		spacedRGB.append(rgb)
	return spacedRGB


def calculate_lightness(l, r, g, b):
	l = (l - 1)  # reverse confidence
	l = ((l - 0) / (0.1 - 0)) * (0.02 - 0.1) + 0  # normalise
	# l = l/100 # turn into percentage

	tmp_r = (255 - r) * l + r
	tmp_g = (255 - g) * l + g
	tmp_b = (255 - b) * l + b

	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


# rgb operates on additive colour channels
def mix_two_rgb(r1, g1, b1, r2, g2, b2):
	tmp_r = (r1 * 0.5) + (r2 * 0.5)
	tmp_g = (g1 * 0.5) + (g2 * 0.5)
	tmp_b = (b1 * 0.5) + (b2 * 0.5)

	if tmp_r > 255:
		tmp_r = r
	if tmp_b > 255:
		tmp_b = b
	if tmp_g > 255:
		tmp_g = g

	return int(math.trunc(tmp_r)), math.trunc(tmp_g), math.trunc(tmp_b)


def get_BG_colour():
	rgb = [BG_colour, BG_colour, BG_colour]
	return tuple(rgb)


# wheel gradient
def get_wheel_gradient(x, y, cx, cy):
	rx = x - cx
	ry = y - cy
	h = ((math.atan2(ry, rx) / math.pi) + 2.17) / 2.0
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


# wheel segment
def get_wheel_segment(x, y, cx, cy):
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


# ring gradient
def get_ring_gradient(x, y, cx, cy):
	rx = x - cx
	ry = y - cy
	h = np.sqrt(rx ** 2.0 + ry ** 2.0)
	h = h / (np.sqrt(2) * min(x_axis, y_axis) / 2)
	#if h <= 0.88: sny die circle kleiner
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])
	#else:
	#	return tuple([255,255,255])


# ring segment
def get_ring_segment(x, y, cx, cy):
	radius = min(x_axis, y_axis) / 2
	cx = cx - 0.5
	cy = cy - 0.5

	num_rings = 5 # change this for more rings
	num_colours = num_rings + 1
	increment = round(radius / num_rings) + 0.1
	hue_increment = 360 / num_colours

	rx = x - cx
	ry = y - cy
	s = round(np.sqrt(rx ** 2.0 + ry ** 2.0))

	# determine the quadrant/segment in which the distance from the centre lies as a point
	if s <= increment * num_rings:
		for i in range(num_colours):
			if s >= increment * i and s < increment * (i + 1):
				h = hue_increment * i
	else:
		h = hue_increment * num_rings

	h = h / 360  # normalise [0,1]
	rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)
	return tuple([int(round(c * 255.0)) for c in rgb])


# plot: wheel gradient
def plot_wheel_gradient():
	im = np.full((y_axis, x_axis, 3), BG_colour)  # 3D array with black BG
	cx, cy = im.shape[1] / 2, im.shape[0] / 2
	for x in range(im.shape[1]):
		for y in range(im.shape[0]):
			im[y, x] = get_wheel_gradient(x, y, cx, cy)
	im = np.array(im, dtype=np.uint8)
	return im


# plot: wheel segment
def plot_wheel_segment():
	im = np.full((y_axis, x_axis, 3), BG_colour)  # 3D array with black BG
	cx, cy = im.shape[1] / 2, im.shape[0] / 2
	for x in range(im.shape[1]):
		for y in range(im.shape[0]):
			im[y, x] = get_wheel_segment(x, y, cx, cy)
	# im = np.array(im, dtype=np.uint8)
	return im


# plot: ring gradient
def plot_ring_gradient():
	im = np.full((y_axis, x_axis, 3), BG_colour)  # 3D array with black BG
	cx, cy = im.shape[1] / 2, im.shape[0] / 2
	for x in range(im.shape[1]):
		for y in range(im.shape[0]):
			im[y, x] = get_ring_gradient(x, y, cx, cy)
	# im = np.array(im, dtype=np.uint8)
	return im


# plot: ring segment
def plot_ring_segment():
	im = np.full((y_axis, x_axis, 3), BG_colour)  # 3D array with black BG
	cx, cy = im.shape[1] / 2, im.shape[0] / 2
	for x in range(im.shape[1]):
		for y in range(im.shape[0]):
			im[y, x] = get_ring_segment(x, y, cx, cy)
	# im = np.array(im, dtype=np.uint8)
	# plot_image("get_ring_segment", im)
	return im


def get_dot_coordinates(x, y):
	# arr_out = []
	arr_out = [(x, y)]
	return arr_out


def get_diamond_coordinates(x, y, d):
	arr_out = []

	## Create diamond shape
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
	arr_out = [(x, y)]

	## Create cross shape
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


def get_circle_coordinates(x, y, d):
	temp_arr = []

	## Create a circle
	# circumference (outline)
	for t in range(0, 360, 1):
		my_x = x + round(d * math.cos(t))
		my_y = y + round(d * math.sin(t))
		# create borders
		if my_x > x_axis - 1:
			my_x = x_axis - 1
		if my_x < 0:
			my_x = 0
		if my_y > y_axis - 1:
			my_y = y_axis - 1
		if my_y < 0:
			my_y = 0

		temp_arr.append((int(my_x), int(my_y)))

	# remove duplicates
	temp_arr = list(set(temp_arr))

	# circle's inner pixels
	circumferenc_dict = {}
	# tuple: Tuple[int, int]
	for tuple in temp_arr:
		val_x, val_y = tuple
		if val_y in circumferenc_dict:
			circumferenc_dict[val_y].append(val_x)
		else:
			temp_list = [val_x]
			circumferenc_dict[val_y] = temp_list

	arr_circle = []
	for val_y in circumferenc_dict:
		my_max = max(circumferenc_dict[val_y])
		my_min = min(circumferenc_dict[val_y])
		if my_min == my_max:
			if my_min < x_axis / 2:
				my_min = 0
			else:
				my_max = x_axis - 1
		for i in range(my_min, my_max + 1):  # include max value
			arr_circle.append((i, val_y))

	return arr_circle


def plot_image(title, img):
	# pyplot.figure(figsize=(20, 4))
	pyplot.imshow(img)
	pyplot.title(title)
	pyplot.axis('off')
	pyplot.show()


def get_coordinates_shape(x, y, d=2):
	# return get_dot_coordinates(x, y)
	# return get_diamond_coordinates(x, y, d)
	return get_crosshair_coordinates(x, y, d)
	# return get_circle_coordinates(x, y, d)


def plot_rgb_coordinates(image, coordinates, colour):
	r, g, b = colour
	for x, y in coordinates:
		r0, g0, b0 = image[y, x]
		if r0 == g0 == b0 == BG_colour:
			image[y, x] = (r, g, b)  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = mix_two_rgb(r, g, b, r0, g0, b0)
	#plot_image("plot mask", image)
	return image


def plot_monochrome_coordinates(image, coordinates, mono_shade):
	for x, y in coordinates:
		m0 = image[y, x]
		if m0 == BG_colour:
			image[y, x] = mono_shade  # x & y coordinates are swapped (Python represents 3D array differently)
		else:  # blend the two colours
			image[y, x] = ((m0 + mono_shade) / 2)
	return image


def impose_mask_on_image(fg, mask):
	for x in range(mask.shape[1]):
		for y in range(mask.shape[0]):
			if mask.ndim == 3:
				if not mask[y, x][0] == mask[y, x][1] == mask[y, x][2] == BG_colour:
					fg[y, x] = mask[y, x]
			else:
				if not mask[y, x] == BG_colour:
					fg[y, x] = mask[y, x]
	return fg


# based on the rule of thirds, the image is centrally cropped
def crop_image(image, centroid):
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

	if image.ndim == 3:
		image = image[int(y_start):int(y_end), int(x_start):int(x_end), :]
	else:
		image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
	return image


def generate_representation(df, centroid_coor, fg_rgb, fg_depth, datatype):
	# extract necessary values from dataframe
	cor_x = df.filter(regex="x\d+").astype(int)
	cor_y = df.filter(regex="y\d+").astype(int)
	cor_conf = df.filter(regex="c\d+").round(6)
	cl = df[[' class (number)']]
	im_filename = df.iloc[0, 0]

	tmp_rgb_mask = np.full((y_axis, x_axis, 3), BG_colour, np.uint8)  # will be used to blend coordinate colours and then imposed on rgb image
	tmp_mono_mask = np.full((y_axis, x_axis), BG_colour, np.uint8)  # will be used to blend coordinate colours and then imposed on rgb image
	spacedRGB = get_spaced_rgb_colours(cor_x.shape[1])  # get an evenly distributed array of colours
	# for every coordinate in the set : thus every col
	if cl.iloc[0, 0] != 6:  # images without any person in the frame - no joints to map, render only a blank representations image
		for col in range(0, cor_x.shape[1]):
			x = int(cor_x.iloc[0, col])
			y = int(cor_y.iloc[0, col])
			confidence = cor_conf.iloc[0, col]

			# if the joint is detected (non-occluded landmark)
			if not (x == 0 and y == 0 and confidence == 0):
				if datatype == 1: # joint colour
					r, g, b = spacedRGB[col]
				if datatype == 2:  # wheel gradient
					r, g, b = get_wheel_gradient(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 3:  # wheel segment
					r, g, b = get_wheel_segment(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 4:  # ring gradient
					r, g, b = get_ring_gradient(x, y, centroid_coor[0], centroid_coor[1])
				if datatype == 5:  # ring segment
					r, g, b = get_ring_segment(x, y, centroid_coor[0], centroid_coor[1])

				# plot joint as a shape represented as a set of coordinates
				xy_coordinates = get_coordinates_shape(x, y)
				tmp_rgb_mask = plot_rgb_coordinates(tmp_rgb_mask, xy_coordinates, [r, g, b])
				tmp_mono_mask = plot_monochrome_coordinates(tmp_mono_mask, xy_coordinates, abs(BG_colour - 255))

	# display the masks
	# cv2.imshow("RGB mask", tmp_rgb_mask)
	# cv2.imshow("MONO mask", tmp_mono_mask)
	# cv2.waitKey(1)
	rgb_out = impose_mask_on_image(fg_rgb, tmp_rgb_mask)
	mono_out = impose_mask_on_image(fg_depth, tmp_mono_mask)

	if save_samples:
		cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[datatype]), "Samples/rgb_mask/"+im_filename), cv2.cvtColor(tmp_rgb_mask, cv2.COLOR_RGB2BGR))
		cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[datatype]), "Samples/rgb_imposed/" + im_filename), cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
		cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[datatype]), "Samples/depth_mask/" + im_filename), tmp_mono_mask)
		cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[datatype]), "Samples/depth_imposed/" + im_filename), mono_out)

	return rgb_out, mono_out


def generate_agumentation_illustrattions(dir):
	bg_im = plot_wheel_gradient()
	Image.fromarray(np.array(bg_im, dtype=np.uint8)).save(dir + '/' + (data_types[1]) + ".png")

	bg_im = plot_wheel_segment()
	Image.fromarray(np.array(bg_im, dtype=np.uint8)).save(dir + '/' + (data_types[2]) + ".png")

	bg_im = plot_ring_gradient()
	Image.fromarray(np.array(bg_im, dtype=np.uint8)).save(dir + '/' + (data_types[3]) + ".png")

	bg_im = plot_ring_segment()
	Image.fromarray(np.array(bg_im, dtype=np.uint8)).save(dir + '/' + (data_types[4]) + ".png")



if __name__ == "__main__":

	# initialise variables
	x_axis, y_axis = 160, 120  # dimensions of input video frame (the image will be resized to this specification)
	x_crop, y_crop = 156, 108  # crop video frame to this dimension
	BG_colour = 0  # 0: black; 255: white
	data_types = list(["0_Baseline", "1_JointColour", "2_RadialGradient", "3_RadialSegment", "4_RingGradient", "5_RingSegment"])

	# initialise dirs
	data_dir = os.path.join(os.getcwd(), "data")
	video_frames_dir = os.path.join(data_dir, "1_video-frames")  # invidual images (video frames)
	keypoints_dir = os.path.join(data_dir, "2_openpose-keypoints", str(x_axis) + "x" + str(y_axis))  # csv files
	generated_dir = os.path.join(data_dir, "3_generated-data-RGBA", str(x_crop) + "x" + str(y_crop))  # generated images
	augmentation_illustration_dir = os.path.join(output_dir, "augmentation-illustration")  # colour wheel illustrations


	# META
	display_frames = False  # show the subtracted frames - will not generate new data, only display
	display_fixed_BG_SUB = True
	display_adaptive_BG_SUB = True
	# depends on BG SUB flags
	display_masks = True

	save_samples = True  # save the generated frames
	# Individual processes
	type_0 = True
	type_1 = True
	type_2 = True
	type_3 = True
	type_4 = True
	type_5 = True

	# create directories
	for folder in list(data_types):
		parent_folder = os.path.join(os.path.join(output_dir, folder), "image-samples")
		create_directories([os.path.join(parent_folder, "RGB_keypoints-overlayed"), os.path.join(parent_folder, "RGB_keypoints"), os.path.join(parent_folder, "DEPTH_keypoints-overlayed"), os.path.join(parent_folder, "DEPTH_keypoints")])
		# create Image dir alongside Samples
		for subfolder in ["empty", "lying", "sitting", "crawling", "bending", "standing"]:
			create_directories([os.path.join(os.path.join(os.path.join(output_dir, folder), "Validation"), subfolder)])
			create_directories([os.path.join(os.path.join(os.path.join(output_dir, folder), "Training"), subfolder)])
			create_directories([os.path.join(os.path.join(os.path.join(output_dir, folder), "Testing"), subfolder)])

	create_directories([augmentation_illustration_dir])
	generate_agumentation_illustrattions(augmentation_illustration_dir)

	datagen_stats_file = open(output_dir + "/DataGen_DatasetStatistics.txt", "w+")  # stores stats on the number of classes in dataset
	datagen_skipframes_file = open(output_dir + "/DataGen_SkippedFrames.txt", "w+")

	for data_partition in list(['Training', 'Validation', 'Testing']):
		print("\n### Generating data partition:\t" + data_partition + " ###\n")

		# prepare output CSV files (flat array of 3D image pixel arrays)
		csv_files = [open(os.path.join(os.path.join(output_dir, dataset_name), data_partition + ".csv"), 'w+') for dataset_name in list(data_types)]

		# import key points from CSV files
		keypoint_input_csv = os.path.join(input_dir, "Keypoints_" + data_partition + ".csv")
		df_orig = pd.read_csv(keypoint_input_csv)
		df_orig['Filename'] = df_orig['Filename'].str.replace('_keypoints.json', '')  # remove unwanted part of filename
		stats.write("DATA PARTITION: " + data_partition)
		print_stats(df_orig, stats)

		# get the set numbers of the dataset e.g. Training_{1172} : pertains to a specific participant and room
		image_dataset_dir = os.path.join(video_frames_dir, data_partition)
		frame_sets = [os.path.basename(f).split('_')[1] for f in df_orig['Filename'].tolist()]

		# iterate through each set to allow simplified BG subtraction
		frame_sets = set(frame_sets)
		skipped.write("\n" + data_partition.upper() + "\n")
		for set_num in frame_sets:  # ITERATE THROUGH EACH SET

			# FIXED BG SUBTRACTION #
			bg_rgb = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "bg"), set_num + "_rgb.png"))
			bg_depth = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "bg"), set_num + "_depth.png"), cv2.IMREAD_GRAYSCALE)

			# ADAPTIVE BG SUBTRACTION #
			frame_history = 1000
			subtractor_rgb = cv2.createBackgroundSubtractorMOG2(history=frame_history, varThreshold=25, detectShadows=True)
			subtractor_depth = cv2.createBackgroundSubtractorMOG2(history=frame_history, detectShadows=False)

			# Identify the image sets (collections of participant and room specific instances)
			filter = data_partition + "_" + set_num
			df_tmp = df_orig[df_orig['Filename'].str.contains(filter)]
			frame_indexes = [os.path.basename(f).split('_')[2] for f in df_tmp['Filename'].tolist()]
			frame_indexes.sort(key=int)

			# loop through video frames and apply background subtraction for each set
			if display_frames:
				for idx in frame_indexes:  # iterate through each set

					# filter data frame to single image
					filter = data_partition + "_" + set_num + "_" + str(idx)
					df = df_tmp[df_tmp['Filename'].str.contains(filter)]
					im_filename = df.iloc[0, 0]
					im_filetype = im_filename.split('_')[0]

					# GET CURRENT FRAME'S RGB & DEPTH IMAGE
					im_rgb = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "rgb"), im_filename))
					im_depth = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "depth"), im_filename), cv2.IMREAD_GRAYSCALE)
					# show original rgb and depth frames
					cv2.imshow('RGB Frame', im_rgb)
					cv2.imshow('DEPTH Frame', im_depth)

					# FIXED BACKGROUND SUBTRACTION #
					if display_fixed_BG_SUB:
						# RGB #
						rgb_mask, rgb_fg = fixed_BG_SUB(im_rgb, bg_rgb)
						if display_masks:
							cv2.imshow('Fixed: RGB MASK', rgb_mask)
						cv2.imshow('Fixed: RGB FOREGROUND', rgb_fg)
						# DEPTH #
						depth_mask, depth_fg = fixed_BG_SUB(im_depth, bg_depth)
						if display_masks:
							cv2.imshow('Fixed: DEPTH MASK', depth_mask)
						cv2.imshow('Fixed: DEPTH FOREGROUND', depth_fg)

					# FIXED BACKGROUND SUBTRACTION #
					if display_adaptive_BG_SUB:
						# RGB #
						rgb_mask, rgb_fg = adaptive_BG_SUB(im_rgb)
						if display_masks:
							cv2.imshow('Adaptive: RGB MASK', rgb_mask)
						cv2.imshow('Adaptive: RGB FOREGROUND', rgb_fg)
						# DEPTH #
						depth_mask, depth_fg = adaptive_BG_SUB(im_depth)
						if display_masks:
							cv2.imshow('Adaptive: DEPTH MASK', depth_mask)
						cv2.imshow('Adaptive: DEPTH FOREGROUND', depth_fg)

					key = cv2.waitKey(1)

			else:
				for idx in frame_indexes:  # iterate through each set

					# Each image will be processed in various ways to compile a number of datasets
					# The data is output to separate csv files where the rgb and depth layers are merged into a 4D array

					# sift through the dataframe for relevant poses/images
					# NOTE: 160x120 has its openpose coordinates in this range, but original images are in the original resolution
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
					count_col = cor_x.shape[1]  # gives number of col count

					# skip the frame if it has too few valid joints (severe occlusion or false detection)
					if cl.iloc[0, 0] != 6:  # if not a deliberately empty frame
						valid_pts = 0
						for col in range(0, cor_conf.shape[1]):
							if cor_conf.iloc[0, col] > 0.0:
								valid_pts += 1
						if valid_pts <= 13:
							skipped.write(im_filename + "\n")
							continue

					# Body centroid used in cropping and centroid-based representations
					# the body centroid is determined as the midpoint between the neck (1), R-hip (8), and L-hip (11) coordinates
					count = 0
					centroid_x = x_axis / 2  # default
					centroid_y = y_axis / 2  # default
					constant_valid_pts = 13
					if cl.iloc[0, 0] != 6:  # if not a empty class representation (no joints), or false/inadequate detection
						# neck is not detected, use closest alternative
						if not cor_conf.iloc[0, 1] > 0.0:  # neck
							if cor_conf.iloc[0, 0] > 0.0:  # nose
								centroid_x = int(cor_x.iloc[0, 0])
								centroid_y = int(cor_y.iloc[0, 0])
								count += 1
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
							count += 1

						# midhip is not detected
						if not cor_conf.iloc[0, 8] > 0.0:
							if cor_conf.iloc[0, 9] > 0.0:  # RHip
								centroid_x += int(cor_x.iloc[0, 9])
								centroid_y += int(cor_y.iloc[0, 9])
								count += 1
							elif cor_conf.iloc[0, 12] > 0.0:  # LHip
								centroid_x += int(cor_x.iloc[0, 12])
								centroid_y += int(cor_y.iloc[0, 12])
								count += 1
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
							count += 1

						# average to obtain midpoint
						if count > 0:
							centroid_x = centroid_x / count
							centroid_y = centroid_y / count
						else:
							# the default midpoint is the half of axis lengths (see variables at start of code block)
							print("\t!!! Reverted to default centroid for image: " + str(df.iloc[0, 0]))
					centroid_coor = (int(centroid_x), int(centroid_y))

					# GET CURRENT FRAME'S RGB & DEPTH IMAGE #
					print(str(data_partition + "  " + set_num + "  " + idx))
					im_rgb = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "rgb"), im_filename))
					im_depth = cv2.imread(os.path.join(os.path.join(image_dataset_dir, "depth"), im_filename), cv2.IMREAD_GRAYSCALE)
					# scale the images from original size to the specified resizing dimensions
					bg_rgb = cv2.resize(bg_rgb, (x_axis, y_axis))
					bg_depth = cv2.resize(bg_depth, (x_axis, y_axis))
					im_rgb = cv2.resize(im_rgb, (x_axis, y_axis))
					im_depth = cv2.resize(im_depth, (x_axis, y_axis))

					# BACKGROUND SUBTRACTION #
					# RGB images uses Adaptive subtraction; Depth images uses Fixed subtraction
					#fg_rgb = cv2.cvtColor(fixed_BG_SUB(im_rgb, bg_rgb)[1], cv2.COLOR_BGR2RGB)  # convert bgr format to standard rgb
					fg_rgb = adaptive_BG_SUB(im_rgb)[1]
					fg_depth = fixed_BG_SUB(im_depth, bg_depth)[1]


					## PRINT DATA SETS TO FILE ##
					VERBOSE_RGB = False
					VERBOSE_DEPTH = False
					CSV = False
					RGBA = True


					# DATASET 0: Foreground images / Baseline (RGB directly stacked onto depth without augmentation)
					dataset_id = 0

					out_im = stack_layers(crop_image(fg_rgb, centroid_coor), crop_image(fg_depth, centroid_coor)) # directly stack rgb and depth images
					if CSV:  # Write to CSV
						out_string = stack_toString(cl.values[0], out_im)  # add class and flatten
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]),
											   im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)
					if save_samples:  # sample images
						cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[dataset_id]), "image-samples/rgb/" + im_filename), cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR))
						cv2.imwrite(os.path.join(os.path.join(output_dir, data_types[dataset_id]), "image-samples/depth/" + im_filename), fg_depth)


					# DATASET 1: Joint colour
					dataset_id = 1
					im_rgb_out, im_mono_out = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_mono_out, centroid_coor))
					if VERBOSE_RGB:  # flick switch for verbose generation
						cv2.imshow(data_types[dataset_id] + ": rgb", cv2.cvtColor(crop_image(im_rgb_out, centroid_coor), cv2.COLOR_RGB2BGR))
					if VERBOSE_DEPTH:
						cv2.imshow(data_types[dataset_id] + ": mono", crop_image(im_mono_out, centroid_coor))
					if CSV:  # Write to CSV
						out_string = stack_toString(cl.values[0], out_im)
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]),
											   im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)


					# DATASET 2: Wheel gradient
					dataset_id = 2
					im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]),
										   im_filetype + "/" + im_class + "/" + im_filename)
					m_rgb_out, im_mono_out = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_mono_out, centroid_coor))
					if VERBOSE_RGB:  # flick switch for verbose generation
						cv2.imshow(data_types[dataset_id] + ": rgb", cv2.cvtColor(crop_image(im_rgb_out, centroid_coor), cv2.COLOR_RGB2BGR))
					if VERBOSE_DEPTH:
						cv2.imshow(data_types[dataset_id] + ": mono", crop_image(im_mono_out, centroid_coor))
					if CSV: # write to file
						out_string = stack_toString(cl.values[0], out_im)
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)


					# DATASET 3: Wheel segment
					dataset_id = 3
					im_rgb_out, im_mono_out = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_mono_out, centroid_coor))
					if VERBOSE_RGB:  # flick switch for verbose generation
						cv2.imshow(data_types[dataset_id] + ": rgb", cv2.cvtColor(crop_image(im_rgb_out, centroid_coor), cv2.COLOR_RGB2BGR))
					if VERBOSE_DEPTH:
						cv2.imshow(data_types[dataset_id] + ": mono", crop_image(im_mono_out, centroid_coor))
					if CSV:  # write to file
						out_string = stack_toString(cl.values[0], out_im)
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]), im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)


					# DATASET 4: Ring gradient
					dataset_id = 4
					im_rgb_out, im_mono_out = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_mono_out, centroid_coor))
					if VERBOSE_RGB:  # flick switch for verbose generation
						cv2.imshow(data_types[dataset_id] + ": rgb", cv2.cvtColor(crop_image(im_rgb_out, centroid_coor), cv2.COLOR_RGB2BGR))
					if VERBOSE_DEPTH:
						cv2.imshow(data_types[dataset_id] + ": mono", crop_image(im_mono_out, centroid_coor))
					if CSV:  # write to file
						out_string = stack_toString(cl.values[0], out_im)
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]), im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)


					# DATASET 5: Ring segment
					dataset_id = 5
					m_rgb_out, im_mono_out = generate_representation(df, centroid_coor, fg_rgb, fg_depth, dataset_id)
					out_im = stack_layers(crop_image(im_rgb_out, centroid_coor), crop_image(im_mono_out, centroid_coor))
					if VERBOSE_RGB:  # flick switch for verbose generation
						cv2.imshow(data_types[dataset_id] + ": rgb", cv2.cvtColor(crop_image(im_rgb_out, centroid_coor), cv2.COLOR_RGB2BGR))
					if VERBOSE_DEPTH:
						cv2.imshow(data_types[dataset_id] + ": depth", crop_image(im_mono_out, centroid_coor))
					if CSV:  # write to file
						out_string = stack_toString(cl.values[0], out_im)
						csv_files[dataset_id].write(out_string + "\n")
					if RGBA:  # Write RGBA Image
						im_path = os.path.join(os.path.join(output_dir, data_types[dataset_id]), im_filetype + "/" + im_class + "/" + im_filename)
						new_image = Image.fromarray(out_im)
						new_image.save(im_path)



					if VERBOSE_RGB:
						cv2.waitKey(1)
					if VERBOSE_DEPTH:
						cv2.waitKey(1)


					gc.collect()
				gc.collect()