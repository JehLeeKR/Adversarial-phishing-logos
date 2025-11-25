import os

import numpy as np
import torch

from classification import (Discriminator, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5, evaluate_by_class

# ------------------------------------------------------#
#   test_annotation_path    Test image path and labels
# ------------------------------------------------------#
test_annotation_path = 'test_data.txt'
# ------------------------------------------------------#
#   metrics_out_path        Folder to save metrics
# ------------------------------------------------------#
metrics_out_path = "metrics_out"


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
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    discriminator = Evaluator()

    with open(test_annotation_path, "r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(discriminator, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1 * 100))
    print("top-5 accuracy = %.2f%%" % (top5 * 100))
    print("mean Recall = %.2f%%" % (np.mean(Recall) * 100))
    print("mean Precision = %.2f%%" % (np.mean(Precision) * 100))

if __name__ == "__main__":
    main()
