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


def plot_image(img, window_title, fig_title):
	""" Simple plot image function that takes a 3D array of RGB pixel values """
	pyplot.figure(figsize=(4, 4))
	pyplot.title(fig_title)
	pyplot.imshow(img)
	man = pyplot.get_current_fig_manager()
	man.canvas.set_window_title(window_title)

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