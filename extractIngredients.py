import os
import json

from externals.ingredientExtractor.ingredient_phrase_tagger.training import utils

BASE_DIR = os.path.dirname(__file__)
METADATA = BASE_DIR + "Yummly28K/metadata27638"
tmpFile = BASE_DIR + "tmp_ingredients.txt"
tmpExtracted = BASE_DIR + "tmp_extracted.txt"
outputFile = BASE_DIR + "extracted_ingredients.json"
outputFilteredFile = BASE_DIR + "new_filtered_ingredients.json"
ingredientsFile = BASE_DIR + "filtered_few.json"
outputIngredientsFile = BASE_DIR + "filtered_few_correct.json"
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
    count = 0
    ingredientFreq = {}
    with open(outputFile, "r") as f:
        ingredients = json.load(f)
    with open(ingredientsFile, "r") as f:
        ingredientsList = json.load(f)
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
            if "powder" not in i:
                if i in ingredientsList:
                    setIngredients.add(i)
                    ingredientFreq.setdefault(i,0)
                    ingredientFreq[i]+=1
                elif i[:-1] in ingredientsList:
                    setIngredients.add(i[:-1])
                    ingredientFreq.setdefault(i[:-1],0)
                    ingredientFreq[i[:-1]]+=1
                elif i+"s" in ingredientsList:
                    setIngredients.add(i+"s")
                    ingredientFreq.setdefault(i+"s",0)
                    ingredientFreq[i+"s"]+=1
                else:
                    if "green" not in i:
                        if "onion" in i or "shallot" in i:
                            ing = "onions"
                            setIngredients.add(ing)
                            ingredientFreq.setdefault(ing,0)
                            ingredientFreq[ing]+=1
                        elif "bean" in i :
                            ing = "beans"
                            setIngredients.add(ing)
                            ingredientFreq.setdefault(ing,0)
                            ingredientFreq[ing]+=1
                    if "lettuce" in i:
                        ing = "lettuce"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "squash" in i:
                        ing = "squash"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "egg" in i:
                        ing = "eggs"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "bell pepper" in i:
                        ing = "bell pepper"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "pancetta" in i:
                        ing = "bacon"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "broth" in i or "stock" in i:
                        ing = "broth"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "potato" in i:
                        ing = "potato"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "pasta" in i or "noodle" in i:
                        ing = "noodles"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "steak" in i or "beef" in i:
                        ing = "beef"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "tomato" in i or "ketchup" in i:
                        ing = "tomato"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "corn" in i:
                        ing = "corn"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "cilantro" in i:
                        ing = "cilantro"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "chicken" in i:
                        ing = "chicken"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "flour" in i:
                        ing = "flour"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "bread" in i:
                        ing = "bread"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "jalapeno" in i:
                        ing = "jalapeno"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "rice" in i:
                        ing = "rice"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "pork" in i:
                        ing = "pork"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "spinach" in i:
                        ing = "spinach"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "turkey" in i:
                        ing = "turkey"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "avocado" in i:
                        ing = "avocado"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1
                    elif "radish" in i:
                        ing = "radish"
                        setIngredients.add(ing)
                        ingredientFreq.setdefault(ing,0)
                        ingredientFreq[ing]+=1

        newMap[filename] = [i for i in setIngredients]
        if (len(newMap[filename])>1):
            count+=1
        
    print("count", count)
    with open(outputFilteredFile,"w") as f:
        json.dump(newMap,f)
    with open(outputIngredientsFile,"w") as f:
        json.dump(ingredientFreq,f)

filter_new()


