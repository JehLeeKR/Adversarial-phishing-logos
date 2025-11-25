# Adversarial logos against phishing detection systems: Code repository

Ethical Statement: This work is done for research purposes; the logos thus generated should not be used for any unethical or illegal purpose.
----
## Research Paper

Lee, J., Xin, Z., See, M. N. P., Sabharwal, K., Apruzzese, G., & Divakaran, D. M. (2023, September). Attacking Logo-Based Phishing Website Detectors with Adversarial Perturbations. In *European Symposium on Research in Computer Security* (pp. 162-182).


```bibtex
@inproceedings{lee2023attacking,
  title={Attacking Logo-Based Phishing Website Detectors with Adversarial Perturbations},
  author={Lee, Jehyun and Xin, Zhe and See, Melanie Ng Pei and Sabharwal, Kanav and Apruzzese, Giovanni and Divakaran, Dinil Mon},
  booktitle={European Symposium on Research in Computer Security},
  pages={162--182},
  year={2023}
}
```

---

## Project Structure

This repository contains the code to generate adversarial perturbations against logo-based phishing detectors. The main components are:
- **Generative Adversarial Perturbations (GAP)**: A generator model is trained to produce subtle, image-dependent perturbations that cause logo detection models to fail.
- **Target Models**: The attacks are evaluated against four different models: ViT, Swin Transformer, Siamese, and Siamese+.
- **Evaluation Scripts**: Scripts to measure the effectiveness of the attacks (e.g., fooling ratio, p-hash distance).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JehLeeKR/Adversarial-phishing-logos
    cd Adversarial-phishing-logos
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install the packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Detectron2:**
    Detectron2 has specific installation requirements based on your PyTorch and CUDA versions. Please follow the official Detectron2 installation guide.

    For example, for PyTorch 2.0 and CUDA 11.8:
    ```bash
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

## How to Run

All commands should be run from the `src/` directory.

### Training the Adversarial Perturbation Generator (GAP)

The core training script is `gap_siamese_with_classification.py`. You can train the generator to fool different target models.

**Training against Siamese/Siamese+:**

The script is configured by default to train against a Siamese network. The loss function aims to decrease the similarity score below a target threshold.

```bash
python gap_siamese_with_classification.py --mode train --expname siamese_attack_exp --nEpochs 50 --batchSize 32 --lr 0.0002
```

**Training against ViT or Swin:**

To train the generator against ViT or Swin Transformer models, use the `gap_vit_clipping.py` or `gap_swin_clipping.py` scripts, respectively. These scripts train the generator to cause misclassification by the target model.
You can run them similarly to the Siamese training script, adjusting parameters as needed.

### Evaluating Fooling Ratios

The `eval_fooling_threshold_on_three.py` script is used to evaluate the fooling ratio of a trained generator against all four models.

```bash
python eval_fooling_threshold_on_three.py \
    --checkpoint_customize <path_to_your_trained_generator.pth> \
    --expname metrics_out/fooling_ratio_comparison \
    --models vit swin siamese siamese+
```

This will:
- Load the generator from the specified checkpoint.
- Generate adversarial images from the test set.
- Evaluate the fooling ratio for each specified model across a range of decision thresholds.
- Save plots and a CSV file with the results in the `metrics_out/fooling_ratio_comparison` directory.

### Evaluating Perceptual Hash (p-hash) Distance

To measure the visual similarity between the original and perturbed images, you can use `eval_p_hash.py`. This script calculates the p-hash distance, where a lower distance means the images are more similar.

```bash
python eval_p_hash.py --checkpoint <path_to_your_trained_generator.pth>
```

This script will generate perturbed images and save a CDF plot and a CSV file of the hamming distances to the `result/` directory.
