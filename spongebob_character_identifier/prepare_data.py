import os
import random
import shutil

from PIL import Image

def convert_to_jpg(source_dir, new_folder_name):
    """
    Converts the images in the input folder to .jpg format and returns the path of the new folder.
    
    Args:
        input_folder (str): Path to the original folder containing 10 subfolders.
        new_folder_name (str): Name for the new folder.
    
    Returns:
        str: Path to the new folder containing converted images.
    """
    # Check if input folder exists
    if os.path.exists(source_dir):
        # Create the new folder if it doesn't exist
        output_folder = new_folder_name
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # List all subdirectories in the input folder
        subdirs = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

        for subdir in subdirs:
            subdir_path = os.path.join(source_dir, subdir)
            output_subdir_path = os.path.join(output_folder, subdir)

            # Create corresponding subdirectories in the new folder
            os.makedirs(output_subdir_path, exist_ok=True)

            # List all files in the source subdirectory
            files = os.listdir(subdir_path)

            for idx, filename in enumerate(files):
                # Check if the file is a .DS_Store file and skip it
                if filename == ".DS_Store":
                    continue

                input_path = os.path.join(subdir_path, filename)
                # Create a new filename based on the original folder name and index
                output_filename = f'{subdir}_{idx}.jpg'
                output_path = os.path.join(output_subdir_path, output_filename)

                try:
                    # Open and convert the image to JPG format
                    with Image.open(input_path) as img:
                        img.convert('RGB').save(output_path, 'JPEG')
                    # print(f'Converted {input_path} to {output_path}')
                except Exception as e:
                    print(f'Error converting {input_path}: {str(e)}')
        print("Done converting images to jpg...")
        return output_folder
    else:
        print("The input folder doesn't exist")

def create_train_test_split(source_dir,
                            new_folder_name,
                            train_percent,
                            test_percent):
    """
    Split and copy data from the source directory into train and test sets.
    
    Args:
        source_dir (str): Path to the source directory containing 10 subfolders.
        data_dir (str): Path to the directory where 'Train' and 'Test' folders will be created.
        train_percent (float): Percentage of data to use for training (between 0 and 1).
        test_percent (float): Percentage of data to use for testing (between 0 and 1).
    """
    # Get a list of subdirectories (the 10 folders)
    subdirs = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # Create 'data' directory if it doesn't exist
    os.makedirs(new_folder_name, exist_ok=True)

    train_dir = os.path.join(new_folder_name, 'train')
    test_dir = os.path.join(new_folder_name, 'test')
    
    # Create 'train' and 'test' directories inside 'data'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        train_subdir_path = os.path.join(train_dir, subdir)
        test_subdir_path = os.path.join(test_dir, subdir)

        # Create corresponding subdirectories in 'train' and 'test'
        os.makedirs(train_subdir_path, exist_ok=True)
        os.makedirs(test_subdir_path, exist_ok=True)

        # List all files in the source directory
        files = os.listdir(subdir_path)

        # Calculate the number of files for train and test based on percentages
        num_files = len(files)
        num_train = int(train_percent * num_files)
        num_test = int(test_percent * num_files)

        # Randomly shuffle the files
        random.shuffle(files)

        # Copy the selected number of files to 'Train' and 'Test' directories
        for file in files[:num_train]:
            shutil.copy(os.path.join(subdir_path, file), os.path.join(train_subdir_path, file))
        
        for file in files[num_train:num_train + num_test]:
            shutil.copy(os.path.join(subdir_path, file), os.path.join(test_subdir_path, file))

def convert_and_split_data(source_dir, new_folder_name, train_percent, test_percent):
    """
    Convert images to JPG format and split them into train and test sets.
    
    Args:
        source_dir (str): Path to the source directory containing 10 subfolders.
        data_dir (str): Path to the directory where 'Train' and 'Test' folders will be created.
        train_percent (float): Percentage of data to use for training (between 0 and 1).
        test_percent (float): Percentage of data to use for testing (between 0 and 1).
    """
    # Convert images to JPG format
    converted_data_path = convert_to_jpg(source_dir, 
                                         new_folder_name="original_data_jpg")

    # Split and copy data into train and test sets
    create_train_test_split(source_dir=converted_data_path,
                            new_folder_name=new_folder_name,
                            train_percent=train_percent,
                            test_percent=test_percent)

# convert_and_split_data(source_dir="original_data",
#                        new_folder_name="data",
#                        train_percent=0.8,
#                        test_percent=0.2)