import torch
import data_setup, engine, utils

from pathlib import Path
from timeit import default_timer as timer 
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

# Import data
DATA_NAME = "character_images"
GITHUB_URL = "https://github.com/gulnuravci/spongebob_character_identifier/raw/main/character_images.zip"
data_setup.import_data_from_github(data_name=DATA_NAME, github_raw_url=GITHUB_URL)

# Setup directories
data_dir = Path("data") / DATA_NAME
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# Setup target device
device = utils.setup_target_device()
# Set the device globally
torch.set_default_device(device)

# Set the name of the model to save
MODEL_NAME = "model_efficientnet_b0_third_run"
NOTES = "Using transfer learning with PyTorch pre-trained model -> efficientnet_b0"

# Setup hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Fix for wrong hash error from: https://github.com/pytorch/vision/issues/7744
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

# Setup the model with pretrained weights and send it to the target device
weights = EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
model = efficientnet_b0(weights=weights).to(device=device)

# Summary before freezing "features" section of the model
utils.summarize_model(model=model, 
                                 input_size=(32, 3, 224, 224),
                                 col_names=["input_size", "output_size", "num_params", "trainable"],
                                 col_width=20,
                                 row_settings=["var_names"])

# Get the transforms used to create pretrained weights
auto_transforms = weights.transforms()

# # Plot transforms to visualize
# utils.plot_transformed_images(image_path=data_dir,
#                                          transform=auto_transforms, 
#                                          n=3,
#                                          seed=None)

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               train_transform=auto_transforms,
                                                                               test_transform=auto_transforms,
                                                                               batch_size=BATCH_SIZE,
                                                                               device=device)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False

# Set seeds
torch.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape,
                    bias=True)).to(device)

# Summary after freezing the "features" section of the model and changing the output classifier layer
utils.summarize_model(model=model, 
                                 input_size=(32, 3, 224, 224),
                                 col_names=["input_size", "output_size", "num_params", "trainable"],
                                 col_width=20,
                                 row_settings=["var_names"])

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# # Attempt a forward pass
# image_batch, label_batch = next(iter(train_dataloader))
# model(image_batch.to(device))

# Set seeds
torch.manual_seed(42)

# Start the timer
start_time = timer()

# Setup training and save the results
model_results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model_with_hyperparameters(model=model,
                                                 model_results=model_results,
                                                 target_dir="models", 
                                                 model_name=MODEL_NAME, 
                                                 num_epochs=NUM_EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 hidden_units="N/A",
                                                 learning_rate=LEARNING_RATE,
                                                 image_size="224x224",
                                                 train_dataloader_length=len(train_dataloader), 
                                                 test_dataloader_length=len(test_dataloader),
                                                 notes=NOTES)

# Plot loss curves
utils.plot_loss_curves(results=model_results)

# Plot predictions
utils.plot_predictions(model=model,
                                  test_dir=test_dir,
                                  data_transform=auto_transforms)

# Plot confusion matrix
utils.confusion_matrix(model=model,
                                  test_dataloader=test_dataloader,
                                  test_dir = test_dir,
                                  test_transform=auto_transforms,
                                  device=device)