import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(__file__)
images_dir = BASE_DIR + "resized_alias/"
labels_dir = BASE_DIR + "ingredient_vector/"
def load_datasets(train_percent = .8):
    usable_filenames = sorted(set([i[3:-4] for i in os.listdir(images_dir)]).intersection(set([i[:-4] for i in os.listdir(labels_dir)])))
    images_tensor = []
    labels_tensor = []
    i=0
    for filename in usable_filenames:
        if i<10:
            print("file " + filename, end="\r")
            image_name = images_dir + "img" + filename +".jpg"
            label_name = labels_dir + filename +".out"
            img_tensor = transforms.ToTensor()(Image.open(image_name))
            images_tensor.append(img_tensor)
            #visualize
            # transforms.ToPILImage()(img_tensor).show()
            with open(label_name, "r") as f:
                labels_tensor.append(torch.Tensor(eval(f.read())))
            i+=1
        else:
            break

    full_dataset = data_utils.TensorDataset(torch.stack(images_tensor), torch.stack(labels_tensor))
    train_size = int(train_percent*len(full_dataset))
    train_dataset, test_dataset = data_utils.random_split(full_dataset, [train_size, len(full_dataset)-train_size])

    return train_dataset, test_dataset
    
load_datasets()