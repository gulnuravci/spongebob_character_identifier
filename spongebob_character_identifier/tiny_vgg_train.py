import torch
import data_setup, engine, model_builder, utils

from pathlib import Path
from torchvision import transforms
from timeit import default_timer as timer 

# Import data
DATA_NAME = "character_images"
GITHUB_URL = "https://github.com/gulnuravci/spongebob_character_identifier/raw/main/character_images.zip"
data_setup.import_data_from_github(data_name=DATA_NAME, github_raw_url=GITHUB_URL)

# Setup directories
data_dir = Path("data") / DATA_NAME
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# # Observe directories
# utils.walk_through_dir(data_dir)

# Setup target device
device = utils.setup_target_device()

# Set the name of the model to save
MODEL_NAME = "model_4"
NOTES = "Extra convolution layer (3 total) with even more epoch trained. No data augmentation used."

# Setup hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Create transforms
train_transform = transforms.Compose([
    # Resize image
    transforms.Resize(size=(64, 64)),
    # transforms.TrivialAugmentWide(num_magnitude_bins=10),
    # Turn image into torch.Tensor
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    # Resize image
    transforms.Resize(size=(64, 64)),
    # Turn image into torch.Tensor
    transforms.ToTensor()
])

utils.plot_transformed_images(image_path=data_dir,
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

# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Create model
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Try a forward pass to figure out the shape for the linear layer in the classifier of the model
# image_batch, label_batch = next(iter(train_dataloader))
# model(image_batch.to(device))

# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
start_time = timer()

# Start training with help from engine.py
model_results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             epochs=NUM_EPOCHS,
                             device=device)

# End the timer and print out how long it took
end_time = timer()

# Save the model
utils.save_model_with_hyperparameters(model=model,
                                      model_results=model_results,
                                      target_dir="models",
                                      model_name=MODEL_NAME,
                                      num_epochs=NUM_EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      hidden_units=HIDDEN_UNITS,
                                      learning_rate=LEARNING_RATE,
                                      image_size="64x64",
                                      train_dataloader_length=len(train_dataloader),
                                      test_dataloader_length=len(test_dataloader),
                                      notes=NOTES)

# Summarize model
utils.summarize_model(model=model,
                                 input_size=[1,3,64,64])

# Plot loss curves
utils.plot_loss_curves(results=model_results)

# Plot predictions
utils.plot_predictions(model=model,
                                  test_dir=test_dir,
                                  data_transform=test_transform,
                                  sample_size=9,
                                  device=device)

# Plot confusion matrix
utils.confusion_matrix(model=model,
                                  test_dataloader=test_dataloader,
                                  test_dir = test_dir,
                                  test_transform=test_transform,
                                  device=device)

# Load saved model
# loaded_model = utils.load_model(input_shape=3,
#                             hidden_units=HIDDEN_UNITS,
#                             output_shape=len(class_names),
#                             model_save_path=Path("models")/f"{MODEL_NAME}.pth")