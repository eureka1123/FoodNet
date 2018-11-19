import os
import json

BASE_DIR = os.path.dirname(__file__)
METADATA = BASE_DIR + "Yummly28K/metadata27638/"
output_file = BASE_DIR + "extracted_cuisines.json"
filtered_file = BASE_DIR + "filtered_cuisines.json"


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
	with open(filtered_file, "w") as f:
		json.dump(filtered, f)
# extract_cuisines()

def filter_cuisines():
	filtered = {}
	with open(output_file) as f:
		cuisines = json.load(f)
	for file in os.listdir(METADATA):
		print(file, end="\r")
		with open(METADATA+file) as f:
			recipes = json.load(f)
		types = recipes["attributes"]["cuisine"]
		most_popular = ""
		freq = 0
		for t in types:
			if cuisines[t]>freq:
				freq = cuisines[t]
				most_popular = t
		if "Asian" in types:
			print(types)
			print(most_popular)
		filtered.setdefault(t,0)
		filtered[t]+=1
	# with open(output_file, "w") as f:
		# json.dump(cuisines, f)
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
