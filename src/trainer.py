import torch
import torchvision.transforms as transforms
from model import Custom_yolo
from dataset import VOCDataset
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import Custom_loss_function

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
MODEL_SAVE_PATH = "model/saved_model.pt"

class Compose (object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boundingboxes):
        for t in self.transforms:
            image, boundingboxes = t(image), boundingboxes
        return image, boundingboxes

def trainer():
    # preprocess the data, transform
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    train_dataset = VOCDataset (
        "data/train.csv",
        transform = transform,
        img_dir = IMAGE_DIR,
        label_dir = LABEL_DIR
    )
    # make the data loader to load the data
    train_loader = DataLoader (
        dataset = train_dataset,
        batch_size = 4,
        shuffle = False,
        drop_last = True
    )
    # make the model
    model = Custom_yolo()
    # train the model
    # train 135 epochs on training set as in paper
    # first 75 epochs with 0.01 lr
    epochs = 75
    optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay = 0.0005)
    mean_loss = []
    cus_loss = Custom_loss_function()
    # train loop
    for epoch in range(epochs):
        print(mean_loss)
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = cus_loss(out, y)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("LR 0.01 Training: epoch: {} batch: {}/7".format(epoch, batch_idx))
    # 30 epochs with 0.001 lr
    epochs = 75
    optimizer = optim.Adam(model.parameters(), lr = 0.000001, weight_decay = 0.0005)
    mean_loss = []
    cus_loss = Custom_loss_function()
    # train loop
    for epoch in range(epochs):
        print(mean_loss)
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = cus_loss(out, y)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("LR 0.001 Training: epoch: {} batch: {}/7".format(epoch, batch_idx))
    torch.save(model, MODEL_SAVE_PATH)
    # 30 epochs with 0.0001 lr
    # parameters
    epochs = 75
    optimizer = optim.Adam(model.parameters(), lr = 0.0000001, weight_decay = 0.0005)
    mean_loss = []
    cus_loss = Custom_loss_function()
    torch.save(model, MODEL_SAVE_PATH)
    # train loop
    for epoch in range(epochs):
        print(mean_loss)
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = cus_loss(out, y)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("LR 0.0001 Training: epoch: {} batch: {}/7".format(epoch, batch_idx))
    torch.save(model, MODEL_SAVE_PATH)
    return