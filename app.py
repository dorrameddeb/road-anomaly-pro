import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F_tv
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torchvision.models as models

st.set_page_config(page_title="Détection d'Anomalies", layout="wide")
st.title("🚗 IA de Détection d'Anomalies Routières (BO1)")

device = torch.device("cpu")
norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# On ne charge PLUS les modèles au démarrage. On attend l'image.
uploaded_file = st.file_uploader("📸 Uploadez une image de route", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Chargement de l'IA et analyse en cours..."):
        # --- CHARGEMENT DES MODÈLES (à la demande) ---
        model_dso1 = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
        model_dso1.load_state_dict(torch.load("modele_dso1.pth", map_location=device))
        model_dso1.eval()

        model_dso2 = models.resnet18(weights=None)
        model_dso2.fc = nn.Linear(model_dso2.fc.in_features, 1)
        model_dso2.load_state_dict(torch.load("modele_dso2.pth", map_location=device))
        model_dso2.eval()
        
        # --- DSO1 ---
        img_256 = F_tv.resize(image, (256, 256), antialias=True)
        tensor_256 = norm(F_tv.to_tensor(img_256)).unsqueeze(0).to(device)
        with torch.no_grad():
            out_mask = torch.sigmoid(model_dso1(tensor_256))
            mask_np = (out_mask > 0.5).float().cpu().squeeze().numpy()
            
        # --- Filtre anti-bruit ---
        mask_clean = (mask_np * 255).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Image Originale", use_column_width=True)
            
        with col2:
            if mask_clean.sum() == 0:
                st.image(np.array(img_256), caption="Aucune anomalie", use_column_width=True)
                st.success("✅ ROUTE DÉGAGÉE - Aucun danger.")
            else:
                mask_final = (mask_clean > 0).astype(np.float32)
                img_224 = F_tv.resize(image, (224, 224), antialias=True)
                tensor_224 = norm(F_tv.to_tensor(img_224)).unsqueeze(0).to(device)
                with torch.no_grad():
                    danger_ratio = model_dso2(tensor_224).item()
                    
                img_orig_np = np.array(img_256)
                overlay = img_orig_np.copy()
                overlay[mask_final == 1] = [255, 0, 0]
                img_finale = cv2.addWeighted(img_orig_np, 0.6, overlay, 0.4, 0)
                
                st.image(img_finale, caption="🔍 Détection (Rouge)", use_column_width=True)
                st.write(f"**Surface occupée :** {danger_ratio*100:.2f}%")
                
                if danger_ratio > 0.15:
                    st.error("🚨 DANGER ÉLEVÉ (High) - Freinage d'urgence !")
                elif danger_ratio > 0.05:
                    st.warning("⚠️ DANGER MOYEN (Mid) - Ralentir.")
                else:
                    st.success("✅ DANGER FAIBLE (Low) - Surveillance.")
                    
        # --- LIBÉRATION DE LA MÉMOIRE ---
        del model_dso1, model_dso2
        torch.cuda.empty_cache() # (vide la mémoire si GPU, inoffensif sur CPU)