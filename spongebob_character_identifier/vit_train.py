import torch
import data_setup, utils, vit, engine

from torch import nn
from pathlib import Path
from torchvision import transforms
from torchinfo import summary
from timeit import default_timer as timer 

# ViT paper link: https://arxiv.org/abs/2010.11929

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
MODEL_NAME = "ViT-Base"
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

# Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
vit_model = vit.ViT(num_classes=len(class_names))

# Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
summary(model=vit_model,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# Setup the optimizer to optimize the ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(params=vit_model.parameters(),
                             lr=3e-3, # Base LR From Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Start the timer
start_time = timer()

# Train the model
model_results = engine.train(model=vit_model,
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
utils.save_model_with_hyperparameters(model=vit_model,
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
utils.plot_predictions(model=vit_model,
                       test_dir=test_dir,
                       data_transform=manual_transforms)

# Plot confusion matrix
utils.confusion_matrix(model=vit_model,
                       test_dataloader=test_dataloader,
                       test_dir = test_dir,
                       test_transform=manual_transforms,
                       device=device)
