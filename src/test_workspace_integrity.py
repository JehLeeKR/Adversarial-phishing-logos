import os
import sys
import argparse
import subprocess
import unittest
import shutil
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import config

# Check if gdown is available at the module level to use with skipIf
try:
    subprocess.run(['gdown', '--version'], check=True, capture_output=True, text=True)
    gdown_installed = True
except (subprocess.CalledProcessError, FileNotFoundError):
    gdown_installed = False


class TestWorkspaceIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary workspace with dummy files for testing."""
        print("\n--- Setting up dummy files for testing ---")
        cls.dummy_dir = "classification/dummy_dataset"
        cls.dummy_logs = "classification/dummy_logs"
        cls.dummy_classes = "classification/dummy_classes.txt"

        # Dummy directories for txt_annotation
        os.makedirs(os.path.join(cls.dummy_dir, "train/class1"), exist_ok=True)
        os.makedirs(os.path.join(cls.dummy_dir, "test/class1"), exist_ok=True)
        Image.new('RGB', (10, 10)).save(os.path.join(cls.dummy_dir, "train/class1/img1.jpg"))
        Image.new('RGB', (10, 10)).save(os.path.join(cls.dummy_dir, "test/class1/img2.jpg"))

        # Dummy class file
        with open(cls.dummy_classes, "w") as f:
            f.write("class1\n")
            
        # Create dummy classes.txt in the path expected by config.py
        cls.config_classes_path = config.CLASSES_PATH
        os.makedirs(os.path.dirname(cls.config_classes_path), exist_ok=True)
        shutil.copy(cls.dummy_classes, cls.config_classes_path)

        # Dummy model and weights for Discriminator
        os.makedirs(cls.dummy_logs, exist_ok=True)
        dummy_model = torch.nn.Linear(1, 1)
        torch.save(dummy_model.state_dict(), os.path.join(cls.dummy_logs, "dummy_weights.pth"))
        print("✅ Dummy files created.")

    @classmethod
    def tearDownClass(cls):
        """Remove all dummy files and directories created during the test setup."""
        print("\n--- Cleaning up dummy files ---")
        try:
            shutil.rmtree(cls.dummy_dir)
            shutil.rmtree(cls.dummy_logs)
            os.remove(cls.dummy_classes)
            if os.path.exists(cls.config_classes_path):
                os.remove(cls.config_classes_path)
            if os.path.exists("train_data.txt"):
                os.remove("train_data.txt")
            if os.path.exists("test_data.txt"):
                os.remove("test_data.txt")
            print("✅ Cleanup complete.")
        except OSError as e:
            print(f"⚠️  Warning: Could not clean up all dummy files: {e}")

    def test_imports(self):
        """Checks if all required third-party libraries can be imported."""
        print("\n--- Checking required packages ---")
        try:
            import matplotlib
            import pandas
            from thop import profile
            from torchsummary import summary
            import imagehash
            print("✅ All third-party packages are installed.")
        except ImportError as e:
            self.fail(f"❌ Import Error: {e}. Please install the missing package.")

    def test_summary_script(self):
        """Tests the classification/summary.py script."""
        print("\n--- Testing: classification/summary.py ---")
        # Temporarily add the classification directory to the path to resolve the 'nets' import
        classification_path = os.path.join(os.path.dirname(__file__), 'classification')
        with patch.dict('sys.modules', {'nets': MagicMock()}):
             with patch.object(sys, 'path', sys.path + [classification_path]):
                from classification.summary import get_args, main
                # Simulate command-line arguments
                args = argparse.Namespace(backbone='resnet50', num_classes=10, input_shape=[224, 224])
                # We mock the main function as we only want to test the import
                with patch('classification.summary.main') as mock_main:
                    print("✅ summary.py imported successfully.")

    def test_txt_annotation_script(self):
        """Tests the classification/txt_annotation.py script."""
        print("\n--- Testing: classification/txt_annotation.py ---")
        # Mock the get_classes function to avoid reading the real file
        # It should return our dummy class name and a count.
        with patch('classification.txt_annotation.get_classes', return_value=(['class1'], 1)):
            # Import the module *inside* the patch context so the mock is active during import.
            from classification import txt_annotation

            # Override global variables in the script for the test
            txt_annotation.classes_path = self.dummy_classes
            txt_annotation.datasets_path = self.dummy_dir
            
            txt_annotation.main()
            
            # Verify output files were created
            self.assertTrue(os.path.exists("train_data.txt"))
            self.assertTrue(os.path.exists("test_data.txt"))
        print("✅ txt_annotation.py executed successfully.")

    def test_classification_module(self):
        """Tests the classification/classification.py module."""
        print("\n--- Testing: classification/classification.py ---")
        # We patch `load_state_dict` to prevent it from trying to load our dummy weights
        # into a real ResNet model, which would cause a key mismatch error.
        with patch('torch.nn.Module.load_state_dict', return_value=None):
            from classification.classification import Discriminator
            # Instantiate the class with dummy paths. The model loading is now mocked.
            discriminator = Discriminator(
                model_path=os.path.join(self.dummy_logs, "dummy_weights.pth"),
                swin_model_path=os.path.join(self.dummy_logs, "dummy_weights.pth"),
                classes_path=self.dummy_classes,
                backbone='resnet50' # Use a standard, simple backbone for the test
            )
            # Create a dummy image to test detection functions
            dummy_image = Image.new('RGB', (224, 224))
            with patch('matplotlib.pyplot.show') as mock_show:
                preds = discriminator.detect_image_with_clipping(dummy_image, clipping=1.0)
                self.assertIsInstance(preds, np.ndarray)
        print("✅ classification.py (Discriminator class) instantiated and tested successfully.")

    @unittest.skipIf(not gdown_installed, "`gdown` is not installed. Skipping model download test. Please run `pip install gdown`.")
    def test_download_and_setup_models(self):
        """
        Checks for and downloads required model files from Google Drive.
        This test ensures that the project has all necessary assets to run.
        """
        print("\n--- Checking for required model and data files ---")

        # Define files to download: (file_id, destination_path, is_zip)
        files_to_download = [
            ("1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH", config.RCNN_MODEL_PATH, False),
            ("1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW", config.RCNN_CONFIG_PATH, False),
            ("1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS", config.SIAMESE_MODEL_PATH, False),
            ("1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I", config.EXPAND_TARGETLIST_ZIP_PATH, True),
            ("1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1", config.DOMAIN_MAP_PATH, False),
            ("1iiAshVLEwsgZH_awbi9Wa6b4JvX_KBE1", config.PHISHPEDIA_MODEL_PATH, False),
        ]

        for file_id, dest_path, is_zip in files_to_download:
            # Create the destination directory if it doesn't exist
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)

            if not os.path.exists(dest_path):
                print(f"⏳ Downloading {os.path.basename(dest_path)}...")
                try:
                    # Use gdown to download the file
                    subprocess.run(['gdown', file_id, '-O', dest_path], check=True)
                    print(f"✅ Downloaded {os.path.basename(dest_path)} successfully.")

                    # If the file is a zip, extract it
                    if is_zip:
                        print(f"🤐 Unzipping {os.path.basename(dest_path)}...")
                        unzip_dir = dest_path.replace('.zip', '')
                        shutil.unpack_archive(dest_path, unzip_dir)
                        print(f"✅ Unzipped to {unzip_dir}")

                except subprocess.CalledProcessError as e:
                    self.fail(f"❌ Failed to download {os.path.basename(dest_path)}. Error: {e}")
            else:
                print(f"✅ {os.path.basename(dest_path)} already exists.")
        
        print("✅ All required model files are present.")


    def test_complex_script_imports(self):
        """
        Performs a basic import test on scripts with data dependencies.
        We patch the file-loading functions to avoid errors from missing data files,
        allowing us to check for syntax and other import errors.
        """
        print("\n--- Testing complex scripts (Import and Syntax Check) ---")
        
        # These scripts fail on import because they try to load data files.
        # We will patch the problematic file-loading functions.
        scripts_to_test = [
            "eval_p_hash_siamese",
            "gap_swin_clipping",
            "gap_vit_clipping",
            "phishpedia.siamese_eval_util",
            "phishpedia.siamese_eval_iden",
            "phishpedia.siamese_eval_tp",
        ]


        # Patch 'gz_pickle_load' and 'read_list' which are used to load data files.
        # We make them return a MagicMock object instead of trying to read from disk.
        with patch('gap_util.gz_pickle_load', MagicMock(return_value=["dummy_file_list"])), \
             patch('phishpedia.siamese_eval_util.read_list', MagicMock(return_value=["dummy_list"])):
            for script_name in scripts_to_test:
                with self.subTest(script=script_name):
                    __import__(script_name)
                    print(f"✅ {script_name}.py imported successfully.")

if __name__ == "__main__":
    # Add the 'src' directory to the Python path to resolve local imports
    # This allows running the test script from the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # The 'src' directory itself also needs to be in the path for relative imports
    src_path = os.path.abspath(os.path.dirname(__file__))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    unittest.main()