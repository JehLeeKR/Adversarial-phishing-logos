import os
from PIL import Image
from typing import List, Any
import argparse
import pickle
import numpy as np
import torch
from classification.classification import Discriminator
from utils.utils import (cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5, evaluate_by_class

import matplotlib.pyplot as plt

class Evaluator(Discriminator):
    """
    An evaluator class that extends the Discriminator to measure identification rates.
    """
    def detect_image(self, image: Image.Image, clipping: float) -> np.ndarray:
        """
        Detects and classifies an image with output clipping.

        Args:
            image (Image.Image): The input image.
            clipping (float): The value to divide logits by before softmax.

        Returns:
            np.ndarray: An array of prediction probabilities for each class.
        """
        # Convert image to RGB
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   Pass the image into the network for prediction
            # ---------------------------------------------------#
            preds = torch.softmax(torch.div(self.model(photo)[0], clipping), dim=-1).cpu().numpy()

        return preds
    
    def evaluate_tp(self, tp_test_annotation_path: str, clipping: float, 
                    starting_threshold: float, ending_threshold: float, stride: float) -> List[float]:
        """
        Evaluates the true positive (identification) rate at various thresholds.

        Args:
            tp_test_annotation_path (str): Path to the test annotation file.
            clipping (float): The output clipping value.
            starting_threshold (float): The initial probability threshold.
            ending_threshold (float): The final probability threshold.
            stride (float): The step to increment the threshold.

        Returns:
            List[float]: A list of identification rates for each threshold.
        """
        images: List[Image.Image] = []
        labels: List[int] = []
        tp_rate: List[float] = []
            
        # evaluate TP
        with open(tp_test_annotation_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_file_path = 'classification/' + line.strip().split(";")[1]
                label = int(line.split(';')[0])
                images.append(Image.open(image_file_path))
                labels.append(label)
 
        threshold: float = starting_threshold
        while threshold <= ending_threshold:
            true_positive: int = 0
            print('Current threshold: ' + str("%.2f" % threshold))
        
            for idx, image in enumerate(images):
                # Get prediction probabilities
                preds = self.detect_image(image, clipping)
                probability: float = np.max(preds)

                # If probability is above threshold, count as a true positive
                if probability >= threshold:
                    true_positive += 1

            accuracy: float = (true_positive / len(images))
            print('Identification rate: ' + str("%.5f" % accuracy))
            tp_rate.append(accuracy)
            threshold += stride
            
        return tp_rate
            
        
        

parser = argparse.ArgumentParser(description='evaluate TP against FP')
parser.add_argument('--output_clipping', help='output clipping', type=float, default=2.5)
parser.add_argument('--starting_threshold', help='starting threshold', type=float, default=0.995)
parser.add_argument('--ending_threshold', help='ending threshold', type=float, default=0.998)
parser.add_argument('--stride', help='stride', type=float, default=0.002)
parser.add_argument('--tp_test_annotation_path', help='tp test annotation path', type=str, default='classification/test_data.txt')
parser.add_argument('--phishpedia_tp_rates', help='phishpedia tp', type=str, default='classification/phishpedia_data/step_relu/identification_rates_final.txt')

# Read list to memory
def read_list(file_path: str) -> Any:
    """
    Reads a Python object from a binary file using pickle.

    Args:
        file_path (str): The path to the file.
    Returns:
        Any: The object loaded from the file.
    """
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data
    

# write list to binary file
def write_list(data: Any, file_path: str) -> None:
    """
    Writes a Python object to a binary file using pickle.

    Args:
        data (Any): The Python object to write.
        file_path (str): The path to the file.
    """
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')

def main() -> None:
    """
    Main function to run the identification rate evaluation.
    """
    opt: argparse.Namespace = parser.parse_args()
    # Extract parameters from arguments
    clipping = opt.output_clipping
    starting_threshold = opt.starting_threshold
    ending_threshold = opt.ending_threshold
    stride = opt.stride
    tp_test_annotation_path = opt.tp_test_annotation_path
    phishpedia_tp_file = opt.phishpedia_tp_rates

    # default backbone is vit_b_16
    discriminator: Evaluator = Evaluator() # default backbone is vit_b_16
    # tp_rate_vit = discriminator.evaluate_tp(tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride)
    
    # assert len(tp_rate_vit) == 20
    # write_list(tp_rate_vit, 'metrics_out/identification/vit.txt')
    
    # switch to swin_transformer_small backbone
    discriminator.switch_to_swin()
    tp_rate_swin: List[float] = discriminator.evaluate_tp(tp_test_annotation_path, clipping, 0.975, 0.978, 0.002)
    # tp_rate_swin = discriminator.evaluate_tp(tp_test_annotation_path, clipping, starting_threshold, ending_threshold, stride)
    
    # assert len(tp_rate_swin) == 2
    # tp_rate_old = read_list('metrics_out/identification/swin.txt')
    # tp_rate_swin = tp_rate_old + tp_rate_swin
    # write_list(tp_rate_swin, 'metrics_out/identification/swin.txt')

if __name__ == "__main__":
    main()