import os
import json

from ingredient_phrase_tagger.training import utils

BASE_DIR = os.path.dirname(__file__)
METADATA = BASE_DIR + "Yummly28K/metadata27638"
tmpFile = BASE_DIR + "tmp_ingredients.txt"
tmpExtracted = BASE_DIR + "tmp_extracted.txt"
outputFile = BASE_DIR + "extracted_ingredients.json"

mapFileToIngredients = {}
counter = 0
for filename in os.listdir(METADATA):
    if counter<100:
        with open(os.path.join(METADATA,filename)) as f:
            ingredients = json.load(f)["ingredientLines"]
        with open(tmpFile, 'w') as outfile:
            outfile.write(utils.export_data(ingredients).encode('utf-8'))

        modelPath = "externals/ingredientExtractor/tmp/model_file"
        modelFilename = os.path.join(BASE_DIR, modelPath)
        os.system("crf_test -v 1 -m %s %s > %s" % (modelFilename, tmpFile, tmpExtracted))
        
        mapFileToIngredients[filename] = [i["name"] for i in utils.import_data(open(tmpExtracted)) if "name" in i]
    counter+=1
with open(outputFile, "w") as f:
    json.dump(mapFileToIngredients, f)
# break


