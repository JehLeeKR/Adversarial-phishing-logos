import os
from PIL import Image

import numpy as np
import torch

from classification import (Discriminator, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5, evaluate_by_class

import matplotlib.pyplot as plt

# ------------------------------------------------------#
#   test_annotation_path    Test image path and labels
# ------------------------------------------------------#
test_annotation_path = '../data/others/'

class Evaluator(Discriminator):
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent errors during prediction with grayscale images.
        #   The code only supports prediction for RGB images, all other types of images will be converted to RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   Resize the image without distortion
        # ---------------------------------------------------#
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # ---------------------------------------------------------#
        #   Normalization + add batch_size dimension + transpose
        # ---------------------------------------------------------#
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   Pass the image into the network for prediction
            # ---------------------------------------------------#
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

def main():
    threshold = []
    FP_rate = []

    discriminator = Evaluator()
    files = os.listdir(test_annotation_path)
    starting_thresh = 0.997

    while starting_thresh < 0.999:
        above_threshold = 0
        threshold.append(starting_thresh)
        print(starting_thresh)

        for file in files:
            image_path = test_annotation_path + file
            image = Image.open(image_path)

            preds = discriminator.detect_image(image)
            probability = np.max(preds)

            if probability > starting_thresh :
                above_threshold += 1

        print(above_threshold)
        print(len(files))
        fp = (above_threshold / len(files))
        print(fp)
        FP_rate.append(fp)
        starting_thresh += 0.001

    plt.plot(threshold, FP_rate)
    plt.xlabel('Threshold')
    plt.ylabel('FP rate')
    plt.title('FP against threshold plot')
    plt.savefig('../result/plot/FP_threshold_swin.png')
    plt.show()

if __name__ == "__main__":
    main()
