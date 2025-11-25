import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Union
from torch import nn
import os
from PIL import Image

from .nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)
import config


# --------------------------------------------#
#   When using your own trained model for prediction, you need to modify 3 parameters
#   model_path, classes_path, and backbone all need to be modified!
# --------------------------------------------#
class Discriminator(object):
    """
    A classifier class for detecting and classifying images.
    It can load different models (ViT, Swin Transformer) and perform predictions.
    """

    _defaults = {
        # --------------------------------------------------------------------------#
        #   When using your own trained model for prediction, be sure to modify model_path and classes_path!
        #   model_path points to the weight file in the logs folder, and classes_path points to the txt file under model_data
        #   If a shape mismatch occurs, also pay attention to modifying the model_path and classes_path parameters during training
        # --------------------------------------------------------------------------#
        "model_path": config.VIT_MODEL_PATH,
        "swin_model_path": config.SWIN_MODEL_PATH,
        "classes_path": config.CLASSES_PATH,

        # --------------------------------------------------------------------#
        #   Input image size
        "input_shape": [224, 224],
        # --------------------------------------------------------------------#
        #   Model type used:
        #   resnet50
        #   vit_b_16
        #   swin_transformer_small
        # --------------------------------------------------------------------#
        "backbone": 'vit_b_16',
        # --------------------------------------------------------------------#
        #   This variable controls whether to use letterbox_image for lossless resizing of the input image
        #   Otherwise, perform a CenterCrop on the image
        # --------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   Whether to use Cuda
        #   Can be set to False if there is no GPU
        # -------------------------------#
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n: str) -> Any:
        """
        Retrieves a default value for a given attribute name.

        Args:
            n (str): The name of the attribute.
        """
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize classification
    # ---------------------------------------------------#
    def __init__(self, discriminator_type: str = 'vit', **kwargs: Any):
        """
        Initializes the Discriminator.

        Args:
            discriminator_type (str): The type of discriminator model to use ('vit' or 'swin').
            **kwargs: Additional keyword arguments to override default settings.
        """

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #    Get classes
        # ---------------------------------------------------#
        self.model: nn.Module = None
        self.class_names: List[str]
        self.num_classes: int
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.init_weights()
        if discriminator_type == 'swin':
            self.switch_to_swin()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   Get all classifications
    # ---------------------------------------------------#
    def init_weights(self) -> None:
        """
        Initializes the model and loads the pre-trained weights.
        """
        # ---------------------------------------------------#
        #   Load model and weights
        # ---------------------------------------------------#
        # Initialize model based on backbone
        if self.backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
                                 'swin_transformer_base']:
            self.model = get_model_from_name[self.backbone](num_classes=self.num_classes, pretrained=False)
        else:
            self.model = get_model_from_name[self.backbone](input_shape=self.input_shape, num_classes=self.num_classes,
                                                            pretrained=False)

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval() # Set to evaluation mode
        print('{} model, and classes loaded.'.format(self.model_path))

        # Use CUDA if available
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def switch_to_swin(self) -> None:
        """
        Switches the model to Swin Transformer and re-initializes weights.
        """
        self.backbone = 'swin_transformer_small'
        self.model_path = self.swin_model_path
        self.init_weights()

    # ---------------------------------------------------#
    #   Detect image
    # ---------------------------------------------------#
    def detect_image(self, image: Image.Image) -> str:
        """
        Detects and classifies a single image, then displays it with the predicted class and probability.

        Args:
            image (Image.Image): The input image.

        Returns:
            str: The predicted class name.
        """

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
            photo = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            # ---------------------------------------------------#
            #   Pass the image into the network for prediction
            # ---------------------------------------------------#
            preds: np.ndarray = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        # ---------------------------------------------------#
        #   Get the category it belongs to
        # ---------------------------------------------------#
        class_name: str = self.class_names[np.argmax(preds)]
        probability: float = np.max(preds)

        # ---------------------------------------------------#
        #   Draw and write text
        # ---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' % (class_name, probability))
        plt.show()
        return class_name
    
    def detect_image_with_clipping(self, image: Image.Image, clipping: float) -> np.ndarray:
        """
        Detects and classifies an image with output clipping on logits.

        Args:
            image (Image.Image): The input image.
            clipping (float): The value to divide the logits by before softmax.

        Returns:
            np.ndarray: An array of prediction probabilities for each class.
        """
        # Convert image to RGB
        image = cvtColor(image)
        # Resize image
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        # Preprocess image
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # Get predictions with output clipping
            preds = torch.softmax(torch.div(self.model(photo)[0], clipping), dim=-1).cpu().numpy()

        return preds
    
    def evaluate_fp(self, clipping: float, thresholds: List[float], fp_test_annotation_path: str) -> List[float]:
        """
        Evaluates the false positive rate on a given dataset.

        Args:
            clipping (float): The output clipping value.
            thresholds (List[float]): A list of probability thresholds to test.
            fp_test_annotation_path (str): Path to the directory with false positive test images.

        Returns:
            List[float]: A list of false positive rates corresponding to each threshold.
        """
        fp_rate: List[float] = []
        fp_rate_count: Dict[float, int] = {}
        fp_files: List[str] = os.listdir(fp_test_annotation_path)

        # Initialize counts for each threshold
        for i in thresholds:
            fp_rate_count[i] = 0
            
        total = len(fp_files)
        for file in fp_files:
            if file.startswith('.'):
                total -= 1
                continue
            image_path = fp_test_annotation_path + file
            image: Image.Image = Image.open(image_path)
            
            # Get predictions
            preds = self.detect_image_with_clipping(image, clipping)
            probability: float = np.max(preds)

            # Check against each threshold
            for thres in thresholds:
                if probability >= thres:
                    fp_rate_count[thres] += 1

        # Calculate FP rate for each threshold
        for i in thresholds:
            fp: float = float(fp_rate_count[i]) / float(total)
            fp_rate.append(fp)
            print('FP rate: ' + str("%.5f" % fp))
            
        return fp_rate
