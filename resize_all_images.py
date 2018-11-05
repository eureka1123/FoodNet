import PIL
from PIL import Image
import os

"""
Note: You need the Yummly28K directory (the data) to resize the images

Resizes all images inside Yummly dataset to the specified size, 
currently set to (250, 167)

Creates two directors, resized_alias/ and resized_antialias/ which 
contain the images resized to the target size, but one uses the antialiasing
resampling filter
"""

# Specified target size to resize to
target_size = (250, 167)

# Filepath to data directory
base_filepath = "Yummly28K/images27638/"

# Names of files to output resized images
alias_dir = "resized_alias"
antialias_dir = "resized_antialias"

if alias_dir not in os.listdir():
	os.mkdir(alias_dir)
if antialias_dir not in os.listdir():
	os.mkdir(antialias_dir)

total_files = len(['_' for file in os.listdir(base_filepath) if 'DS' not in file])

print("Total number of files to resize: " + str(total_files))

count = 0
for file in os.listdir(base_filepath):
	count += 1
	print("Image " + str(count), end="\r")
	if "DS" not in file:
		img = Image.open(base_filepath + file)
		size = img.size
		if size is not target_size:
			img_antialias = img.resize(target_size, PIL.Image.ANTIALIAS)
			img_antialias.save(antialias_dir + "/" + file)
			
			img_alias = img.resize(target_size)
			img_alias.save(alias_dir + "/" + file)	