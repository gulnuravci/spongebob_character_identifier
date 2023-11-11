import torch
import data_setup, utils, vit, engine
import torchvision 
from torch import nn
from pathlib import Path
from torchvision import transforms
from torchinfo import summary
from timeit import default_timer as timer 

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
MODEL_NAME = "pre_trained_ViT"
NOTES = "N/A"

# Create image size (from Table 3 in the ViT paper)
IMG_SIZE = 224
# Set the batch size
BATCH_SIZE = 32 # this is lower than the ViT paper

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=manual_transforms,
    test_transform=manual_transforms,
    batch_size=BATCH_SIZE,
    device=device
)

# Get pretrainted weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# Change the classifier head
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
summary(model=pretrained_vit,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# Setup the optimizer to optimize the ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Start the timer
start_time = timer()

# Train the model
model_results = engine.train(model=pretrained_vit,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             epochs=10,
                             device=device,
                             writer=utils.create_writer(experiment_name="original",
                                                 model_name=MODEL_NAME,
                                                 extra=f"{10}_epochs"))

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model_with_hyperparameters(model=pretrained_vit,
                                      model_results=model_results,
                                      target_dir="models",
                                      model_name=MODEL_NAME,
                                      num_epochs=10,
                                      batch_size=BATCH_SIZE,
                                      hidden_units="N/A",
                                      learning_rate="1e-3",
                                      image_size="224x224",
                                      train_dataloader_length=len(train_dataloader),
                                      test_dataloader_length=len(test_dataloader),
                                      notes=f"Training time:{end_time-start_time}")

# Plot loss curves
utils.plot_loss_curves(results=model_results)

# Plot predictions
utils.plot_predictions(model=pretrained_vit,
                       test_dir=test_dir,
                       data_transform=manual_transforms)

# Plot confusion matrix
utils.confusion_matrix(model=pretrained_vit,
                       test_dataloader=test_dataloader,
                       test_dir = test_dir,
                       test_transform=manual_transforms,
                       device=device)
