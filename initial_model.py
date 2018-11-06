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

NUM_LABELS = 2240
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_labels=NUM_LABELS
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=2)
        self.fc1 = nn.Linear(41*62*20, 10000)
        self.fc2 = nn.Linear(10000, NUM_LABELS)

    def forward(self, x):
        print("x",x.size())
        # print("conv1",self.conv1(x).size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #input 2, 3, 167, 250
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #size 2, 10, 83, 125
        print(x.size())
        print("after",x.size()) 
        x = x.view(-1, 41*62*20) # 41, 62
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print("batch_idx", batch_idx, data, target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        target=torch.LongTensor(np.array(target.numpy(),np.long))
        data_var=torch.autograd.Variable(data)
        target_var=torch.autograd.Variable(target)

        output = model(data_var)
        # loss_function = nn.MultiLabelMarginLoss()
        loss = nn.MultiLabelMarginLoss()(output,target_var)
        # print(output)
        # print(target_var)
        # loss = F.nll_loss(output, target)

        # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct = pred.eq(target.view_as(pred)).sum().item()
        # sum_num_correct += correct
        # sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
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
            output = model(data)
            test_loss += nn.MultiLabelMarginLoss(output, target, reduction='sum').item() # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

def training_procedure(train_dataset, test_dataset):
    args = dict()
    args["seed"] = 73912
    args["no_cuda"] = False
    args["log_interval"] = 100
    args["batch_size"] = 2
    args["test-batch-size"] = 1000

    params = dict()
    params["epochs"] = 10
    params["lr"] = 0.1

    torch.manual_seed(args["seed"])
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args["batch_size"], shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args["test-batch-size"], shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params["lr"])

    # Train the model
    for epoch in range(1, params["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    train_dataset, test_dataset= load_datasets()
    training_procedure(train_dataset, test_dataset)




