import torch
import data_setup, engine, utils, transfer_learning_model_builder

from pathlib import Path
from timeit import default_timer as timer 
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b2, EfficientNet_B0_Weights, EfficientNet_B2_Weights
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
device = utils.setup_target_device(device="cpu")

# Set the name of the model to save
MODEL_NAME = "model_efficientnet_b2"
NOTES = "Using transfer learning with PyTorch pre-trained model 'efficientnet_b2'"

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OUT_FEATURES = 10

# Create model
effnetb2 = transfer_learning_model_builder.create_effnetb2(out_features=OUT_FEATURES,
                                                           device=device)

# Get transform
weights = EfficientNet_B2_Weights.DEFAULT
auto_transforms = weights.transforms()

# Plot transforms to visualize
utils.plot_transformed_images(image_path=data_dir,
                              transform=auto_transforms,
                              n=3,
                              seed=None)

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               train_transform=auto_transforms,
                                                                               test_transform=auto_transforms,
                                                                               batch_size=BATCH_SIZE,
                                                                               device=device)

# Summary after freezing the "features" section of the model and changing the output classifier layer
utils.summarize_model(model=effnetb2,
                      input_size=(32, 3, 224, 224),
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20,
                      row_settings=["var_names"])

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(effnetb2.parameters(),
                             lr=LEARNING_RATE)

# # Attempt a forward pass
# image_batch, label_batch = next(iter(train_dataloader))
# model(image_batch.to(device))

# Set seeds
torch.manual_seed(42)

# Start the timer
start_time = timer()

# Setup training and save the results
model_results = engine.train(model=effnetb2,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             epochs=NUM_EPOCHS,
                             device=device,
                             writer=utils.create_writer(experiment_name="original",
                                                 model_name=MODEL_NAME,
                                                 extra=f"{NUM_EPOCHS}_epochs"))

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model_with_hyperparameters(model=effnetb2,
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
utils.plot_predictions(model=effnetb2,
                       test_dir=test_dir,
                       data_transform=auto_transforms)

# Plot confusion matrix
utils.confusion_matrix(model=effnetb2,
                       test_dataloader=test_dataloader,
                       test_dir = test_dir,
                       test_transform=auto_transforms,
                       device=device)