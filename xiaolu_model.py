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
labels_dir = BASE_DIR + "ingredient_vector_threshold100/"
dict_file= BASE_DIR + "ingredientnames.out"
freq_file= BASE_DIR + "filtered_100.json"

# NUM_LABELS = 2240
NUM_LABELS = 373

usable_filenames = sorted(set([i[3:-4] for i in os.listdir(images_dir)]).intersection(set([i[:-4] for i in os.listdir(labels_dir)])))

with open(dict_file,"r") as f:
    listIngredients = eval(f.read())
with open(freq_file,"r") as f:
    freqIngredients = json.load(f)
POS_WEIGHT = torch.Tensor([(len(usable_filenames)-freqIngredients[i])/freqIngredients[i] for i in listIngredients])

#100x149
#50x74
#25x37
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_labels=NUM_LABELS
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5,padding=2)
        # self.conv3 = nn.Conv2d(40, 60, kernel_size=5,padding=2)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=5,padding=2)
        self.fc1 = nn.Linear(25*37*40, 1000)
        # self.fc1 = nn.Linear(41*62*40, 1000)
        self.fc2 = nn.Linear(1000, NUM_LABELS)

    def forward(self, x):
        print("x",x.size())
        # print("conv1",self.conv1(x).size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #input 2, 3, 167, 250
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #size 2, 10, 83, 125
        print(x.size())
        # x = F.relu(F.max_pool2d(self.conv3(x), 2)) #size 2, 10, 20, 31
        # print(x.size())
        print("after",x.size()) 
        x = x.view(-1, 25*37*40) 

        # x = x.view(-1, 41*62*40) 
        # x = x.view(-1, 40*31*128) 

        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc2(x)
        print("done",x.size()) 
        # return F.log_softmax(x, dim=1)
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
        if batch_idx %5 == 0:
            for batch in range(len(target)):
                # if batch%10 == 0:
                print("target")
                print([listIngredients[i] for i in range(len(target[batch])) if target[batch][i] == 1])
                print("output")
                # print(output[batch])
                print([i for i in torch.topk(pred[batch], 10, largest = True)])
                print([listIngredients[i] for i in torch.topk(pred[batch], 10, largest = True)[1]])
                print(len([i for i in pred[batch] if i!=0]))
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target=torch.FloatTensor(np.array(target.numpy(),np.long))
            data_var=torch.autograd.Variable(data)
            target_var=torch.autograd.Variable(target)
            # optimizer.zero_grad()

            output = model(data_var)
            # output = model(data)
            test_loss += nn.BCEWithLogitsLoss()(output, target_var).item() # sum up batch loss

            # test_loss += nn.MultiLabelSoftMarginLoss()(output, target_var).item() # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def load_datasets(train_percent = .8):
    # usable_filenames = sorted(set([i[3:-4] for i in os.listdir(images_dir)]).intersection(set([i[:-4] for i in os.listdir(labels_dir)])))
    images_tensor = []
    labels_tensor = []
    i=0
    transformations = transforms.Compose([
        transforms.Resize(100),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    for filename in usable_filenames:
        if i<10000:
            print("file " + filename, end="\r")
            image_name = images_dir + "img" + filename +".jpg"
            label_name = labels_dir + filename +".out"
            img_tensor = transformations(Image.open(image_name))
            images_tensor.append(img_tensor)
            # visualize
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

def training_procedure(train_dataset, test_dataset):
    args = dict()
    args["seed"] = 73912
    args["no_cuda"] = False
    args["log_interval"] = 100
    args["batch_size"] = 20
    args["test-batch-size"] = 100

    params = dict()
    params["epochs"] = 10
    params["lr"] = 0.01

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




