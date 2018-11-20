import os
import json

BASE_DIR = os.path.dirname(__file__)
METADATA = BASE_DIR + "Yummly28K/metadata27638/"
output_file = BASE_DIR + "extracted_cuisines.json"
filtered_file = BASE_DIR + "filtered_cuisines.json"

OUTPUT_DIR = "cuisines_vector"

if OUTPUT_DIR not in os.listdir():
    os.mkdir(OUTPUT_DIR)


def extract_cuisines():
	cuisines = {}
	for file in os.listdir(METADATA):
		print(file, end="\r")
		with open(METADATA+file) as f:
			recipes = json.load(f)
		types = recipes["attributes"]["cuisine"]
		for t in types:
			cuisines.setdefault(t,0)
			cuisines[t]+=1
	with open(output_file, "w") as f:
		json.dump(cuisines, f)
extract_cuisines()

def filter_cuisines():
	filtered = {}
	with open(output_file) as f:
		cuisines = json.load(f)
	print(cuisines)
	a = list(cuisines.keys())
	for file in os.listdir(METADATA):
		#print(file, end="\r")
		with open(METADATA+file) as f:
			recipes = json.load(f)
		types = recipes["attributes"]["cuisine"]
		#print(types)
		most_popular = ""
		freq = 0
		for t in types:
			if cuisines[t]>freq:
				freq = cuisines[t]
				most_popular = t
		filtered.setdefault(t,0)
		filtered[t]+=1

		onehot = [0] * len(a)
		for i in range(len(a)):
			if a[i] == most_popular:
				onehot[i] = 1
		num = file[4:-5]
		print(BASE_DIR + OUTPUT_DIR + "/" + num + ".out")
		with open(BASE_DIR + OUTPUT_DIR + "/" + num + ".out", "w+") as f:
			f.write(str(onehot))

	with open(filtered_file, "w") as f:
		json.dump(filtered, f)

filter_cuisines()
# with open(output_file) as f:
# 	cuisines = json.load(f)
# filtered = {}
# for i in cuisines:
# 	if i == "Indian" or i == "Chinese" or i == "Thai" or i == "Japanese" or i == "Vietnamese":
# 		filtered.setdefault("Asian",0)
# 		filtered["Asian"]+=cuisines[i]
# 	filtered.setdefault(i,0)
# 	filtered[i] += cuisines[i]
# ordered = sorted([(cuisines[i],i) for i in filtered])[::-1]
# # ordered = sorted([(cuisines[i],i) for i in cuisines])[::-1]
# print(ordered)
