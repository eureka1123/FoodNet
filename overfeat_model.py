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
input_dir = BASE_DIR + "overfeatures/"
labels_dir = BASE_DIR + "new_ingredient_vector_threshold/"
dict_file= BASE_DIR + "ingredientnames.out"
freq_file= BASE_DIR + "filtered_few_correct.json"

# NUM_LABELS = 2240
NUM_LABELS = 47

usable_filenames = sorted(set([i[8:-4] for i in os.listdir(input_dir)]).intersection(set([i[:-4] for i in os.listdir(labels_dir)])))

# with open(dict_file,"r") as f:
#     listIngredients = eval(f.read())
with open(freq_file,"r") as f:
    freqIngredients = json.load(f)
listIngredients = [i for i in freqIngredients]
POS_WEIGHT = torch.Tensor([(len(usable_filenames)-freqIngredients[i])/freqIngredients[i] for i in listIngredients])

#100x149
#50x74
#25x37
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_labels=NUM_LABELS
        self.fc1 = nn.Linear(4096*4, NUM_LABELS)

    def forward(self, x):
        x = self.fc1(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    print("train_loader",len(train_loader))
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx", batch_idx, data, target)
        data_var, target_var = data.to(device), target.to(device)
        # target = torch.unsqueeze(target,1)
        target_var=torch.FloatTensor(np.array(target_var.numpy(),np.long))
        # data_var=torch.autograd.Variable(data)
        # target_var=torch.autograd.Variable(target)
        optimizer.zero_grad()

        output = model(data_var)
            # print([listIngredients[i] for i in range(len(output[batch])) if output[batch][i] == 1])
        # loss_function = nn.MultiLabelMarginLoss()
        #pos_weight = torch.Tensor([200 for i in range(373)])
        loss = nn.BCEWithLogitsLoss(reduction='sum',pos_weight = POS_WEIGHT)(output,target_var)
        pred = torch.sigmoid(output)
        # if batch_idx %5 == 0:
        for batch in range(len(target)):
            # if batch%10 == 0:
            train_target = [listIngredients[i] for i in range(len(target[batch])) if target[batch][i] == 1]
            train_output = [listIngredients[i] for i in torch.topk(pred[batch], 10, largest = True)[1]]
            sum_num_correct+=len([i for i in train_target if i in train_output])/len(train_target)
            # print("train target: ",train_target)
            # print("train output: ",train_output)
            # print("train prob: ",[i for i in torch.topk(pred[batch], 10, largest = True)])
            # print("----------------------------------------------------------------------")
        # print(output)
        # print(target_var)
        # loss = F.nll_loss(output, target)

        # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct = pred.eq(target.view_as(pred)).sum().item()
        # sum_num_correct += correct
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx>-1: #% 100 == 0:
            print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum_loss / num_batches_since_log, 
                100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
            )
            sum_num_correct = 0
            sum_loss = 0
            num_batches_since_log = 0

def test(model, device, test_loader, dataset_name="Test set"):
    model.eval()
    test_loss = 0
    correct = 0
    print("----------------------------------------------------------------------")
    print("TEST")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target=torch.FloatTensor(np.array(target.numpy(),np.long))
            data_var=torch.autograd.Variable(data)
            target_var=torch.autograd.Variable(target)
            # optimizer.zero_grad()

            output = model(data_var)
            # output = model(data)
            pred = torch.sigmoid(output)
            for batch in range(len(target)):
            # if batch%10 == 0:
                test_target = [listIngredients[i] for i in range(len(target[batch])) if target[batch][i] == 1]
                test_output = [listIngredients[i] for i in torch.topk(pred[batch], 10, largest = True)[1]]
                correct+=len([i for i in test_target if i in test_output])/len(test_target)
                print("test target: ",test_target)
                print("test output: ",test_output)
                # print(output[batch])
                print("test prob: ",[i for i in torch.topk(pred[batch], 10, largest = True)])
                print("----------------------------------------------------------------------")

            # test_loss += nn.BCEWithLogitsLoss()(output, target_var).item() # sum up batch loss
            test_loss += nn.BCEWithLogitsLoss(reduction='sum')(output,target_var).item()

            # test_loss += nn.MultiLabelSoftMarginLoss()(output, target_var).item() # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\n{}: Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_datasets(train_percent = .8):
    # usable_filenames = sorted(set([i[3:-4] for i in os.listdir(images_dir)]).intersection(set([i[:-4] for i in os.listdir(labels_dir)])))
    input_tensor = []
    labels_tensor = []
    i=0
    for filename in usable_filenames:
        if i<200:
            print("file " + filename, end="\n")
            input_name = input_dir + "features" + filename +".out"
            label_name = labels_dir + filename +".out"
            with open(label_name, "r") as f:
                label = eval(f.read())
            if sum(label)>2:
                labels_tensor.append(torch.Tensor(label))
                stuff = []
                with open(input_name, "r") as f:
                    r = f.read().split('\n')
                    r = r[1].split(' ')[:-1]
                    stuff = [float(x) for x in r]
                input_tensor.append(torch.Tensor(stuff))
                i+=1
        else:
            break

    full_dataset = data_utils.TensorDataset(torch.stack(input_tensor), torch.stack(labels_tensor))
    train_size = int(train_percent*len(full_dataset))
    train_dataset, test_dataset = data_utils.random_split(full_dataset, [train_size, len(full_dataset)-train_size])

    return train_dataset, test_dataset

def training_procedure(train_dataset, test_dataset):
    args = dict()
    args["seed"] = 73912
    args["no_cuda"] = False
    args["log_interval"] = 100
    args["batch_size"] = 20
    args["test-batch-size"] = 20

    params = dict()
    params["epochs"] = 30
    params["lr"] = 0.0001

    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args["batch_size"], shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args["test-batch-size"], shuffle=True, **kwargs)

    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=params["lr"])
    optimizer = optim.Adam(model.parameters(), lr = params["lr"])
    # optimizer = optim.Adam(model.parameters())

    # Train the model
    for epoch in range(1, params["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    train_dataset, test_dataset= load_datasets()
    training_procedure(train_dataset, test_dataset)




