import os
import json

BASE_DIR = os.path.dirname(__file__)

if "ingredient_vector" not in os.listdir():
    os.mkdir("ingredient_vector")

print_frequencies = False
ingredient_count_threshold = 6

def get_data():

    with open(BASE_DIR + "filtered_ingredients.json") as json_data:
        recipes = json.load(json_data)
        json_data.close()
    all_ingredients = {}

    print(len(recipes))
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
        print(BASE_DIR + "ingredient_vector/" + num + ".out")
        with open(BASE_DIR + "ingredient_vector/" + num + ".out","w+") as f:
             f.write(str(r))
        data.append((x,r))
        return data

print(len(get_data()))
print(get_data()[0:100])