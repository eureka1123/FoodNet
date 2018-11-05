import os
import json

BASE_DIR = os.path.dirname(__file__)

print_frequencies = False

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
        if all_ingredients[x] <= 6:
            continue
        sorted_ingredients.append((x, all_ingredients[x]))
    sorted_ingredients.sort(key=lambda x: -x[1])

    print(len(sorted_ingredients))
    if print_frequencies:
        print(sorted_ingredients)

    ingredient_id = {}
    for i in range(len(sorted_ingredients)):
        ingredient_id[sorted_ingredients[i][0]] = i
    data = []
    for x in recipes:
        r = [0] * len(sorted_ingredients)
        ingredients = recipes[x]
        for y in ingredients:
            if y in ingredient_id:
                r[ingredient_id[y]] = 1
        data.append([x, r])
    return data

print(len(get_data()))
print(get_data()[0:100])