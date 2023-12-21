import os
import random
# Set script directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# Specify the directory path
directory_path = '\data'

# Get a list of folders in the directory
folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

# Check if there are any folders
if folders:
    # Randomly pick a folder
    randomly_selected_folder = random.choice(folders)

    # Display the result
    print(f"Randomly selected folder: {randomly_selected_folder}")
else:
    print("No folders found in the specified directory.")