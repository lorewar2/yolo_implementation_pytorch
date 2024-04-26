import trainer
import evaluator
import util
from dataset import VOCDataset
import torchvision.transforms as transforms

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

def main ():
    # train the model on train.csv save model.pt
    #trainer.trainer()

    # evaluate using model.pt and get prediction and ground truth boxes
    prediction, ground = evaluator.evaluator()

    # get the mAP score
    mAP = evaluator.evaluate_predictions(prediction, ground)
    print(f"Mean Average Precision (mAP) across 20 classes: {mAP}")

    # draw the images with bounding boxes
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    example_dataset = VOCDataset (
        "data/100examples.csv",
        transform = transform,
        img_dir = IMAGE_DIR,
        label_dir = LABEL_DIR
    ) 
    #util.draw_all_images(prediction, ground, 5, example_dataset)
    return

if __name__ == "__main__":
    main()
   