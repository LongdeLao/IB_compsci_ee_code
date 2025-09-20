import kagglehub
import shutil
import os

# Download dataset (default location)
path = kagglehub.dataset_download("kushagrapandya/visdrone-dataset")

# Relative target folder (one level below current working directory)
target_dir = "data"  # NOT "/data"

# Make sure the folder exists
os.makedirs(target_dir, exist_ok=True)

# Move the downloaded file to the relative folder
shutil.move(path, target_dir)

print("Dataset moved to:", os.path.join(target_dir, os.path.basename(path)))
