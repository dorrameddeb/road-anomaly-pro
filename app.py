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

# =====================================================================
# CONFIGURATION DE LA PAGE
# =====================================================================
st.set_page_config(page_title="Détection d'Anomalies Routières", layout="wide")
st.title("🚗 IA de Détection d'Anomalies Routières (BO1)")
st.markdown("Projet Deep Learning - Dataset RoadAnomaly. Uploadez une photo pour détecter les obstacles et calculer le niveau de danger.")

# Sur Render (gratuit), on utilise le CPU
device = torch.device("cpu")

# =====================================================================
# CHARGEMENT DES MODÈLES (Mis en cache pour ne pas ralentir l'app)
# =====================================================================
@st.cache_resource
def load_models():
    # 1. Modèle DSO1 (Segmentation U-Net ResNet34)
    model_dso1 = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model_dso1.load_state_dict(torch.load("modele_dso1.pth", map_location=device))
    model_dso1.eval()
    
    # 2. Modèle DSO2 (Régression ResNet18)
    model_dso2 = models.resnet18(weights=None)
    model_dso2.fc = nn.Linear(model_dso2.fc.in_features, 1)
    model_dso2.load_state_dict(torch.load("modele_dso2.pth", map_location=device))
    model_dso2.eval()
    
    return model_dso1, model_dso2

try:
    model_dso1, model_dso2 = load_models()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles. Avez-vous bien uploadé les fichiers .pth ? Détail : {e}")

# =====================================================================
# INTERFACE UTILISATEUR
# =====================================================================
uploaded_file = st.file_uploader("📸 Uploadez une image de route (format JPG, PNG ou WEBP)", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Analyse de l'image par l'Intelligence Artificielle en cours..."):
        # --- DSO1 : SEGMENTATION ---
        img_256 = F_tv.resize(image, (256, 256), antialias=True)
        tensor_256 = norm(F_tv.to_tensor(img_256)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_mask = torch.sigmoid(model_dso1(tensor_256))
            mask_np = (out_mask > 0.5).float().cpu().squeeze().numpy()
            
        # --- POST-TRAITEMENT (Filtre anti-bruit pour les routes vides) ---
        mask_clean = (mask_np * 255).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # --- AFFICHAGE DES RÉSULTATS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Image Originale", use_container_width=True)
            
        with col2:
            # Si le masque est vide après nettoyage : Route dégagée !
            if mask_clean.sum() == 0:
                st.image(np.array(img_256), caption="Aucune anomalie détectée", use_container_width=True)
                st.success("✅ ROUTE DÉGAGÉE - Aucun danger détecté sur la chaussée.")
            else:
                # --- DSO2 : CALCUL DU DANGER ---
                mask_final = (mask_clean > 0).astype(np.float32)
                
                img_224 = F_tv.resize(image, (224, 224), antialias=True)
                tensor_224 = norm(F_tv.to_tensor(img_224)).unsqueeze(0).to(device)
                with torch.no_grad():
                    danger_ratio = model_dso2(tensor_224).item()
                    
                # Création de l'image avec le masque rouge
                img_orig_np = np.array(img_256)
                overlay = img_orig_np.copy()
                overlay[mask_final == 1] = [255, 0, 0]
                img_finale = cv2.addWeighted(img_orig_np, 0.6, overlay, 0.4, 0)
                
                st.image(img_finale, caption="🔍 Détection de l'IA (Anomalie en rouge)", use_container_width=True)
                
                st.markdown("### 🚦 Système d'Alerte Embarqué (BO1)")
                st.write(f"**Surface occupée par l'obstacle :** {danger_ratio*100:.2f}%")
                
                # --- BO1 : LOGIQUE D'ALERTE ---
                if danger_ratio > 0.15:
                    st.error("🚨 DANGER ÉLEVÉ (High) - Freinage d'urgence recommandé !")
                elif danger_ratio > 0.05:
                    st.warning("⚠️ DANGER MOYEN (Mid) - Ralentir, obstacle proche.")
                else:
                    st.success("✅ DANGER FAIBLE (Low) - Obstacle lointain, surveillance.")