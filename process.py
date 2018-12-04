import os
import json

BASE_DIR = os.path.dirname(__file__)

#-------------------------------------------------------------#
########## CHANGE THIS ACCORDING TO OUTPUTS YOU WANT ##########
#-------------------------------------------------------------#
ingredient_count_threshold = 100
WRITE_INGREDIENT_NAMES = True
print_frequencies = False
OUTPUT_DIR = "new_ingredient_vector_threshold" #+ str(ingredient_count_threshold)
#-------------------------------------------------------------#
########## CHANGE THIS ACCORDING TO OUTPUTS YOU WANT ##########
#-------------------------------------------------------------#

if OUTPUT_DIR not in os.listdir():
    os.mkdir(OUTPUT_DIR)

def get_data():

    with open(BASE_DIR + "new_filtered_ingredients.json") as json_data:
        recipes = json.load(json_data)
        json_data.close()
    all_ingredients = {}

    print("number of recipes: " + str(len(recipes)))
    for x in recipes:
        ingredients = recipes[x]
        for y in ingredients:
            all_ingredients[y] = 0
    for x in recipes:
        ingredients = recipes[x]
        for y in ingredients:
            all_ingredients[y] += 1
    sorted_ingredients = []
    for x in all_ingredients:
        if all_ingredients[x] <= ingredient_count_threshold:
            continue
        sorted_ingredients.append((x, all_ingredients[x]))
    sorted_ingredients.sort(key=lambda x: -x[1])

    if WRITE_INGREDIENT_NAMES:
        ingredientnames = [x[0] for x in sorted_ingredients]
        with open(BASE_DIR + "new_ingredientnames.out", "w+") as f:
            f.write(str(ingredientnames))

    print(len(sorted_ingredients))
    if print_frequencies:
        print(sorted_ingredients)

    ingredient_id = {}
    data = []
    for i in range(len(sorted_ingredients)):
        ingredient_id[sorted_ingredients[i][0]] = i
    for x in recipes:
        r = [0] * len(sorted_ingredients)
        ingredients = recipes[x]
        for y in ingredients:
            if y in ingredient_id:
                r[ingredient_id[y]] = 1

        num = x[4:-5]
        # print(BASE_DIR + OUTPUT_DIR + "/" + num + ".out")
        with open(BASE_DIR + OUTPUT_DIR + "/" + num + ".out","w+") as f:
             f.write(str(r))
        data.append((x,r))
    return data

get_data()