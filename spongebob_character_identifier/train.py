import random
import torch
import data_setup, engine, model_builder, utils, helper_functions

from pathlib import Path
from torchvision import transforms, datasets
# Import data
DATA_NAME = "training_data"
GITHUB_URL = "https://github.com/gulnuravci/spongebob_character_identifier/raw/main/data.zip"
data_setup.import_data_from_github(data_name=DATA_NAME, github_raw_url=GITHUB_URL)

# Setup directories
data_dir = Path("data") / DATA_NAME
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# # Observe directories
# helper_functions.walk_through_dir(data_dir)

# Setup target device
device = helper_functions.setup_target_device()

# Set the name of the model to save
MODEL_NAME = "model_20_epochs"

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Create transforms
train_transform = transforms.Compose([
    # Resize image
    transforms.Resize(size=(64, 64)),
    # Turn image into torch.Tensor
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    # Resize image
    transforms.Resize(size=(64, 64)),
    # Turn image into torch.Tensor
    transforms.ToTensor()
])

helper_functions.plot_transformed_images(image_path=data_dir,
                                  transform=train_transform, 
                                  n=3, 
                                  seed=None)

# Create train and test dataloader and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                              test_dir=test_dir,
                              train_transform=train_transform,
                              test_transform=test_transform,
                              batch_size=BATCH_SIZE)

# Check lengths of train and test dataloader
print(f"Batch size: {BATCH_SIZE}")
print(f"Train dataloader length: {len(train_dataloader)}")
print(f"Test dataloader length: {len(test_dataloader)}", end="\n\n")

# Check a sample data point from dataloader
img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}", end = "\n\n")

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
model_results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             epochs=NUM_EPOCHS,
                             device=device)

# Summarize model
helper_functions.summarize_model(model=model,
                                 input_size=[1,3,64,64])

# Plot loss curves
helper_functions.plot_loss_curves(results=model_results)

# Plot predictions
helper_functions.plot_predictions(model=model,
                                  test_dir=test_dir,
                                  data_transform=test_transform,
                                  sample_size=9,
                                  device="cpu")

# Plot confusion matrix
helper_functions.confusion_matrix(model=model,
                                  test_dataloader=test_dataloader,
                                  test_dir = test_dir,
                                  test_transform=test_transform,
                                  device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=f"{MODEL_NAME}.pth")

# Load saved model
helper_functions.load_model(input_shape=3,
                            hidden_units=HIDDEN_UNITS,
                            output_shape=len(class_names),
                            model_save_path=Path("models")/f"{MODEL_NAME}.pth")