import os
import requests
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def import_data_from_github(data_name, github_raw_url):
    # Setup path to a data folder
    data_path = Path("data/")
    image_path = data_path / data_name

    # Check if image directory already exists
    if image_path.is_dir():
        print(f"{image_path} directory already exists, skipping import...")
    else:
        print(f"{image_path} directory does not exist, importing...")

        # Create the image path directory 
        image_path.mkdir(parents=True, exist_ok=True)

        # Write zip file from github
        zip_file_path = data_path / (data_name + ".zip")
        with open(zip_file_path, "wb") as f:
            request = requests.get(github_raw_url)
            f.write(request.content)

        # Unzip data, excluding __MACOSX folders
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip __MACOSX folders and their contents
                if "__MACOSX" in file_info.filename: continue
                zip_ref.extract(file_info, image_path)

        # Clean up the __MACOSX folder if it was extracted
        macosx_folder = image_path / "__MACOSX"
        if macosx_folder.is_dir():
            for item in macosx_folder.glob("*"):
                if item.is_file(): item.unlink()
                elif item.is_dir(): os.rmdir(item)
            os.rmdir(macosx_folder)
        
        # Remove zip file after unzipping
        os.remove("data/training_data.zip")
        
        print("Import complete.")

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int=0):
    """Creates training and testing DataLoaders.

    Converts the specified training and testing directories into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to the training directory.
        test_dir (str): Path to the testing directory.
        train_transform (transforms.Compose): Transformations to apply to training data.
        test_transform (transforms.Compose): Transformations to apply to testing data.
        batch_size (int): Number of samples per batch in each DataLoader.
        num_workers (int, optional): Number of worker processes for DataLoader.
    
    Returns:
        tuple: A tuple containing the training DataLoader, testing DataLoader, and a list of class names.
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    # Get class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names