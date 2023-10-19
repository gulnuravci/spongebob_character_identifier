import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import model_builder

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
from typing import Dict, List
from torchinfo import summary
from pathlib import Path
from torchvision import datasets, transforms
from PIL import Image

def walk_through_dir(dir_path: Path):
    """
    Explore the contents of the specified directory and create a DataFrame summarizing the subdirectories and files.
    
    Args:
        dir_path (Path): The path to the directory to be explored.
    """
    data = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Ignore .DS_Store files
        filenames = [filename for filename in filenames if filename != '.DS_Store']
        num_directories = len(dirnames)
        num_images = len(filenames)
        data.append([dirpath, num_directories, num_images])
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["Directory", "Num_Directories", "Num_Images"])

    # Print DataFrame to console
    print(df)

def setup_target_device(device: str = None):
    # Setup target device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print current device
    print(f"Current device: {device}", end="\n\n")

    return device

def plot_transformed_images(image_path: Path, 
                            transform: transforms.Compose,
                            n: int = 3,
                            seed: int = None):
    """"
    Selects random images from a path of images and loads/transforms them and plots the original vs the transformed version.

    Args:
        image_path (Path): A path to the directory containing images to be selected and transformed.
        transform (transforms.Compose): A data transformation pipeline.
        n (int, optional): The number of random images to select and plot.
        seed (int, optional): The random seed.
    """
    # Set random seed
    if seed: random.seed(seed)

    # Get a list of image paths matching the specified pattern
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # Randomly select 'n' image paths from the list
    random_image_paths = random.sample(image_path_list, k=n)

    # Create a subplot with 'n' rows and 2 columns for the original and transformed images
    fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(10, 10))
    
    # Iterate through the randomly selected image paths
    for i, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            # Plot the original image in the first column
            ax[i, 0].imshow(f)
            ax[i, 0].set_title(f"Original {image_path.parent.stem}\nSize: {f.size}")
            ax[i, 0].axis(False)

            # Transform the image and plot it in the second column
            transformed_image = transform(f).permute(1, 2, 0) # (C, H, W) -> (H, W, C)
            ax[i, 1].imshow(transformed_image)
            ax[i, 1].set_title(f"Transformed {image_path.parent.stem}\nShape: {transformed_image.shape}")
            ax[i, 1].axis("off")

    # Set the title of the entire figure
    fig.suptitle("Transformed Images", fontsize=16)

    # Display figure
    plt.show()

def summarize_model(model: torch.nn.Module,
                    input_size: list):
    """
    Summarizes the given PyTorch model using torchinfo, displaying information about the model's layers and parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to be summarized.
        input_size (list): A list specifying the input size as [batch_size, num_channels, height, width].
    """
    # Generate a summary of the model
    summary(model, input_size=input_size)

def plot_loss_curves(results: Dict[str, List[float]]):
    """
        Plots training curves of a results dictionary.

        Args:
            results (Dict[str, List[float]]): A dictionary containing training and test results, where keys are strings and values are lists of floats. Expected keys: "train_loss", "test_loss", "train_acc", and "test_acc".
                - "train_loss": A list of training loss values.
                - "test_loss": A list of test loss values.
                - "train_acc": A list of training accuracy values.
                - "test_acc": A list of test accuracy values.
        """
    # Get the loss values of the results dictionary(training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))
    plt.title("Loss Curves")

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Show plot
    plt.show()

def get_test_data(test_dir: str,
                  data_transform: transforms.Compose,
                  sample_size: int,
                  seed: int = None):
    """
    Get a sample of test data along with class names and labels.

    Args:
        test_dir (str): The path to the directory containing test data.
        data_transform (transforms.Compose): A data transformation pipeline.
        sample_size (int): The number of samples to randomly select from the test data.
        seed (int, optional): A fixed random seed for reproducible results.

    Returns:
        tuple: A tuple containing lists of class names, test labels, and test samples.
    """
    # Set a fixed random seed for reproducible results
    random.seed(seed)

    # Create an ImageFolder dataset for testing using the provided directory and data transformation
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)
    
    # Get class names
    class_names = test_data.classes

    # Randomly select samples from the test dataset
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=sample_size):
        test_samples.append(sample)
        test_labels.append(label)
    
    # Return tuple of lists
    return class_names, test_labels, test_samples

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = "cpu"):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logits -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

def plot_predictions(model: torch.nn.Module,
                     test_dir: str,
                     data_transform: transforms.Compose,
                     sample_size: int,
                     device: torch.device = "cpu"):
    """
    Makes predictions using the given model on a list of data samples and visualizes the results.

    Args:
        model (torch.nn.Module): The PyTorch model for making predictions.
        test_dir (str): The directory path containing test images.
        class_names (list): A list of class names for label mapping.
        nrows (int, optional): Number of rows for the subplot grid. Default is 3.
        ncols (int, optional): Number of columns for the subplot grid. Default is 3.
    """
    class_names, test_labels, test_samples = get_test_data(test_dir=test_dir, 
                                                           data_transform=data_transform,
                                                           sample_size=9)
    
    pred_probs = make_predictions(model = model,
                                  data = test_samples,
                                  device = "cpu")

    # Convert prediction probabilities to labels
    pred_classes = pred_probs.argmax(dim=1)

    # Create a subplot for visualization
    plt.figure(figsize=(10, 10))
    plt.title("Predictions vs Truth")
    
    nrows = ncols = int(sample_size ** (1/2))
    for i, sample in enumerate(test_samples):
        # Create subplot
        plt.subplot(nrows, ncols, i + 1)

        # Plot the target image
        plt.imshow(sample.permute(1, 2, 0), cmap="gray")

        # Find the prediction (in text form, e.g., "Sandal")
        pred_label = class_names[pred_classes[i]]

        # Get the truth label (in text form)
        truth_label = class_names[test_labels[i]]

        # Create a title for the plot
        title_text = f"Pred: {pred_label} \nTruth: {truth_label}"

        # Check for equality between pred and truth and change the color of title text
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # Green text if prediction is the same as the truth
        else:
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)

    plt.show()

def confusion_matrix(model: torch.nn.Module,
                          test_dataloader: torch.utils.data.DataLoader,
                          test_dir: list,
                          test_transform: transforms.Compose,
                          device: str = "cpu"):
    
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform)
    class_names = test_data.classes

    # 1. Make predictions with trained model
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions..."):
            # Send the data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilites -> prediction labels
            y_pred = y_logit.argmax(dim=1)
            # Put prediction on CPU for evaluation
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

    # Convert truth labels into a tensor
    y_truth_tensor = torch.tensor(test_data.targets)

    # Setup confusion instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=y_pred_tensor,
                             target=y_truth_tensor)
    
    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), # matplotlib likes working with numpy
                                    class_names=class_names,
                                    figsize=(10, 7))

    # Show plot
    plt.show()

def load_model(input_shape: int,
               hidden_units: int,
               output_shape: int,
               model_save_path: str,
               device: str = "cpu",
               seed: int = None):
    torch.manual_seed(seed)
    loaded_model = model_builder.TinyVGG(
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape)
    
    # Load in the save state_dict()
    loaded_model.load_state_dict(torch.load(f=model_save_path))

    # Send the model to the target device
    loaded_model.to(device)

    # Print model to console
    print(loaded_model)