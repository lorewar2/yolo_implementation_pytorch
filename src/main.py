import torch
import torchvision.transforms as transforms
from model import Yolo
from dataset import VOCDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
matplotlib.interactive(False)

def main():
    model = Yolo()
    train_dataset = VOCDataset(
        "data/100examples.csv",
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR
    )
    for idx, (x, y) in enumerate(train_dataset):
        
        plot_image(x)
        #output = model(x)
        
        break
    return

def plot_image(image):
    im = np.array(image)
    height, width, _ = im.shape
    # Create figure and axes
    fig, ax = plt.subplots(1)
    plt.imshow(im, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()
   