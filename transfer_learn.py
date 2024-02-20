import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary

try:
    from going_modular.going_modular import data_setup, engine
except:
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !rm -rf pytorch-deep-learning
    from going_modular.going_modular import data_setup, engine

device = "cuda" if torch.cuda.is_available() else "cpu"
import os
import zipfile

from pathlib import Path
import requests
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)
    os.remove(data_path / "pizza_steak_sushi.zip")
train_dir = image_path / "train"
test_dir = image_path / "test"

manual_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ,])
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir,transform=manual_transforms, batch_size=32) 
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,test_dir=test_dir,transform=auto_transforms, batch_size=32)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
summary(model=model, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20, row_settings=["var_names"]) 
for param in model.features.parameters():
    param.requires_grad = False
# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
output_shape = len(class_names)
model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(in_features=1280,out_features=output_shape,bias=True)).to(device)
summary(model,input_size=(32, 3, 224, 224), verbose=0,col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20,row_settings=["var_names"])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from timeit import default_timer as timer 
start_time = timer()
results = engine.train(model=model, train_dataloader=train_dataloader,test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=5, device=device)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
from helper_functions import plot_loss_curves
plot_loss_curves(results)
from typing import List, Tuple
from PIL import Image
def pred_and_plot_image(model: torch.nn.Module, image_path: str,class_names: List[str],image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None, device: torch.device=device):      
    img = Image.open(image_path)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    model.to(device)
    model.eval()
    with torch.inference_mode():
      transformed_image = image_transform(img).unsqueeze(dim=0)
      target_image_pred = model(transformed_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list,k=num_images_to_plot)
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model, image_path=image_path, class_names=class_names,image_size=(224, 224))

import requests
custom_image_path = data_path / "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get("images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

pred_and_plot_image(model=model, image_path=custom_image_path,class_names=class_names)


