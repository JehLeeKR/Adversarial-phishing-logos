#--------------------------------------------#
#   This part of the code is only for viewing the network structure, not for testing.
#--------------------------------------------#
import argparse
from typing import List, Tuple
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets import get_model_from_name

def get_args() -> argparse.Namespace:
    """
    Parses and returns command-line arguments for the model summary script.
    """
    parser = argparse.ArgumentParser(description="Print a summary of a neural network model.")
    parser.add_argument('--backbone', type=str, default="mobilenetv2",
                        help="The name of the model backbone to summarize.")
    parser.add_argument('--num_classes', type=int, default=1000,
                        help="The number of output classes for the model.")
    parser.add_argument('--input_shape', type=int, nargs=2, default=[224, 224],
                        help="The height and width of the input image.")
    return parser.parse_args()

def main(opt: argparse.Namespace):
    """
    Calculates and prints the model summary, including GFLOPS and total parameters.

    Args:
        opt (argparse.Namespace): Command-line arguments containing model configuration.
    """
    input_shape: List[int] = opt.input_shape
    num_classes: int = opt.num_classes
    backbone: str = opt.backbone

    # --- Model Initialization ---
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = get_model_from_name[backbone](num_classes=num_classes, pretrained=False).to(device)

    # --- Print Model Architecture Summary ---
    # Print model summary
    summary(model, (3, input_shape[0], input_shape[1]))

    # --- Calculate FLOPs and Parameters ---
    dummy_input: torch.Tensor = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2 is because profile does not count convolution as two operations.
    #   Some papers count convolution as two operations: multiplication and addition. In this case, multiply by 2.
    #   Some papers only consider the number of multiplication operations and ignore addition. In this case, do not multiply by 2.
    #   This code chooses to multiply by 2, referencing YOLOX.
    #--------------------------------------------------------#
    flops: float = flops * 2
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops_str))
    print('Total params: %s' % (params_str))

if __name__ == "__main__":
    # Get command-line arguments
    options = get_args()
    # Run the main function
    main(options)
