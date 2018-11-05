import PIL
from PIL import Image
import os

target_size = (250, 167)
base_filepath = "Yummly28K/images27638/"

alias_dir = "resized_alias"
antialias_dir = "resized_antialias"

os.mkdir(alias_dir)
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