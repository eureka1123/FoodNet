import os
import json

from ingredient_phrase_tagger.training import utils

BASE_DIR = os.path.dirname(__file__)
METADATA = BASE_DIR + "Yummly28K/metadata27638"
tmpFile = BASE_DIR + "tmp_ingredients.txt"
tmpExtracted = BASE_DIR + "tmp_extracted.txt"
outputFile = BASE_DIR + "extracted_ingredients.json"
outputFilteredFile = BASE_DIR + "filtered_ingredients.json"
STOPWORDS = {"fresh","cup","cups","tbsp","tsp","large","small",
            "lb","lbs","oz","pound","ounce",
            "pounds","ounces","tablespoons","tablespoon",
            "teaspoons","teaspoon","grams","gram",
            "tbs","liter","litre","inch","inches","centimeter",
            "centimeters","long","pkg","sliced","g","c","t","kg","ml","tl","ts","gms",
            "gm","qts","qt"}
def extract_from_original():
    mapFileToIngredients = {}
    counter = 0
    for filename in os.listdir(METADATA):
        try:
            with open(os.path.join(METADATA,filename)) as f:
                ingredients = json.load(f)["ingredientLines"]
            with open(tmpFile, 'w') as outfile:
                outfile.write(utils.export_data(ingredients).encode('utf-8'))

            modelPath = "externals/ingredientExtractor/tmp/model_file"
            modelFilename = os.path.join(BASE_DIR, modelPath)
            os.system("crf_test -v 1 -m %s %s > %s" % (modelFilename, tmpFile, tmpExtracted))
            
            mapFileToIngredients[filename] = [i["name"] for i in utils.import_data(open(tmpExtracted)) if "name" in i]
        except:
            print(filename)
            counter+=1
    print("errors",counter)
    with open(outputFile, "w") as f:
        json.dump(mapFileToIngredients, f)

def filter_new():
    newMap = {}
    with open(outputFile, "r") as f:
        ingredients = json.load(f)
    for filename in ingredients:
        setIngredients = set()
        for i in ingredients[filename]:
            i = i.replace(u"\u00a0"," ")
            i = i.replace(u"\u00ee","i")
            i = i.replace(u"\u00f1","n")
            i = i.replace(u"\u00e4","a")
            i = i.replace(u"\u00e8","e")
            i = i.replace(u"\u00ae","")
            i = i.replace("-"," ")
            i = i.replace("/"," ")
            i = ''.join([s for s in i if s.isalpha() or s == ' '])
            words = [word.lower() for word in i.split(" ") if word.lower() not in STOPWORDS]      
            i = " ".join([w for w in words if w!=""])
            setIngredients.add(i)
        newMap[filename] = [i for i in setIngredients]
    with open(outputFilteredFile,"w") as f:
        json.dump(newMap,f)

filter_new()


