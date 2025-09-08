import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import json
from monai.networks.nets import UNet

import PIL.Image as Image

def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
    return pred
def preprocess_image(img):

    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()

    # Si image 3D (multi-canaux), prendre le premier
    if img.ndim == 3:
        img = img[0]

    # Resize à 128x128
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Normalisation
    img = img / 255.0

    # Ajouter channel et batch → shape [1,1,128,128]
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor

# Titre de l'app
st.title("Segmentation Médicale avec UNet")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,   # nb de classes
    channels=(16,32,64,128,256),
    strides=(2,2,2,2),
    num_res_units=2
)


st.subheader(" L'architecture du modèle U-Net")
    # Charger l'image
img1 = Image.open("unet_model.png")
st.write("un modèle UNet de segmentation automatique du cœur à partir d’IRM cardiaques(dataset ACDC Challenge)")
    # Afficher l'image
st.image(img1, caption="U-Net")

    # Charger historique
with open("history.json", "r") as f:
        history = json.load(f)

train_losses = history["train_loss"]
val_losses = history["val_loss"]
val_dice = history["val_dice"]
val_acc = history["val_acc"]

st.title("📊 Résultats d'entraînement UNet")

    # Courbe Loss
st.subheader("Loss (Entraînement vs Validation)")
fig, ax = plt.subplots()
ax.plot(train_losses, label="Train Loss", color="red")
ax.plot(val_losses, label="Validation Loss", color="orange")
ax.set_xlabel("Époques")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

    # Courbe Dice
st.subheader("Dice Score")
st.write("Dice Final : ", f"{val_dice[-1]:.3f}")
fig, ax = plt.subplots()
ax.plot(val_dice, label="Validation Dice", color="blue")
ax.set_xlabel("Époques")
ax.set_ylabel("Dice")
ax.legend()
st.pyplot(fig)


    #image vis
st.subheader("Comparaison entre masque de vérité terrain et prédiction du modèle U-Net")
    # Charger l'image
img = Image.open("result.png")

    # Afficher l'image
st.image(img, caption="IRM cardiaque")
st.title("Choisissez une image IRM cardiaque pour tester le modèle")

uploaded_file = st.file_uploader("Choisissez une image médicale IRM cardiaques...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Charger image


    img = np.array(Image.open(uploaded_file).convert("L"))
    st.image(img, caption="Image originale", use_container_width=True)
    img =preprocess_image(img )

    model.load_state_dict( torch.load("unet_model.pth", map_location="cpu"))
    img_tensor = torch.tensor(img/255.0).float().unsqueeze(0)
    # Prédiction
    pred_mask = predict(model,img)


    img_to_show = img.squeeze()
    if isinstance(img_to_show, torch.Tensor):
        img_to_show = img_to_show.cpu().numpy()
    pred_to_show = torch.argmax(pred_mask, dim=1).squeeze()


    if isinstance(pred_mask, torch.Tensor):
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
    else:
        pred_mask_np = pred_mask



    # GT factice
    gt_mask = np.zeros_like(img_to_show)
    gt_mask[50:150, 80:180] = 1

    # Affichage côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_to_show, cmap="gray")
    axes[0].set_title("Image Originale")
    axes[1].imshow(img_to_show, cmap="gray")
    axes[1].imshow(pred_to_show, alpha=0.5, cmap="Reds")
    axes[1].set_title("Masque Prédit")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig)


    if isinstance(pred_mask, torch.Tensor):
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
    else:
        pred_mask_np = pred_mask


    gt_mask_np = gt_mask

    # Calcul de l'intersection
    intersection = np.sum(pred_mask_np * gt_mask_np)
