import os

# --- Project Root ---
# Assumes this config.py is in the 'src' directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# --- Data & Models Directories ---
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# --- Dataset Paths ---
# Original dataset directory: 'datasets_logo_181'
LOGO_DATASET_DIR = os.path.join(DATA_DIR, 'logo_181')
CLASSES_PATH = os.path.join(LOGO_DATASET_DIR, 'classes.txt')

# Cached features for Siamese network dataloader
# Original directory: 'siamese_mag_10_new'
SIAMESE_FEATURES_DIR = os.path.join(DATA_DIR, 'siamese_features_mag_10')
PREPROCESS_LABELS_TRAIN_PATH = os.path.join(SIAMESE_FEATURES_DIR, 'preprocess_labels_train.txt')
PREPROCESS_SIM_TEST_PATH = os.path.join(SIAMESE_FEATURES_DIR, 'preprocess_sim_test.txt')

# --- Model Paths ---

# Classification models
VIT_MODEL_PATH = os.path.join(MODELS_DIR, 'ViT', 'best_epoch_weights.pth')
SWIN_MODEL_PATH = os.path.join(MODELS_DIR, 'Swin', 'best_epoch_weights.pth')

# --- Phishpedia Paths ---
PHISHPEDIA_DIR = os.path.join(SRC_DIR, 'phishpedia')
PHISHPEDIA_MODELS_DIR = os.path.join(MODELS_DIR, 'phishpedia')

# Detectron2 paths
DETECTRON2_PEDIA_DIR = os.path.join(PHISHPEDIA_DIR, 'detectron2_pedia')
RCNN_CONFIG_PATH = os.path.join(DETECTRON2_PEDIA_DIR, 'configs/faster_rcnn.yaml')
RCNN_MODEL_PATH = os.path.join(PHISHPEDIA_MODELS_DIR, 'detectron2', 'rcnn_bet365.pth')

# Siamese paths
SIAMESE_PEDIA_DIR = os.path.join(PHISHPEDIA_DIR, 'siamese_pedia')
SIAMESE_MODEL_PATH = os.path.join(PHISHPEDIA_MODELS_DIR, 'siamese', 'resnetv2_rgb_new.pth.tar')
PHISHPEDIA_MODEL_PATH = os.path.join(PHISHPEDIA_MODELS_DIR, 'siamese', 'finetune_bit.pth.tar')
DOMAIN_MAP_PATH = os.path.join(PHISHPEDIA_MODELS_DIR, 'siamese', 'domain_map.pkl')
EXPAND_TARGETLIST_ZIP_PATH = os.path.join(PHISHPEDIA_MODELS_DIR, 'siamese', 'expand_targetlist.zip')

# --- Test-related paths ---
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test_data')