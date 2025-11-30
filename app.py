import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_unet.h5", compile=False)
    return model

model = load_model()

# Get expected input size from model
_, H, W, C = model.input_shape   # e.g. (None, 256, 256, 3)
IMG_HEIGHT, IMG_WIDTH, N_CHANNELS = H, W, C

def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    arr = np.array(img).astype(np.float32) / 255.0
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    return arr




# -------------------------
# Predict Mask
# -------------------------
def predict_mask(image_arr):
    # image_arr: (H, W, 3)
    inp = np.expand_dims(image_arr, axis=0)  # (1, H, W, 3)
    pred = model.predict(inp)[0, :, :, 0]    # (H, W)
    return pred

st.title("🛰️ Flood Segmentation using U-Net")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(pil_img, use_container_width=True)

    img_arr = preprocess_image(pil_img)

    st.write("Image shape for model:", img_arr.shape)

    with st.spinner("Predicting..."):
        prob_mask = predict_mask(img_arr)   # float mask

    st.write(f"Prediction min: {prob_mask.min():.4f}, max: {prob_mask.max():.4f}")

    # Show probability heatmap to see if model is doing *anything*
    st.subheader("Probability Mask (0–1)")
    st.image(prob_mask, clamp=True, use_container_width=True)

    # # Threshold slider to experiment
    thr = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05)
    bin_mask = (prob_mask > thr).astype(np.uint8)    # 0 or 1

    st.subheader("Binary Predicted Mask")
    st.image(bin_mask * 255, use_container_width=True)

    # Overlay on original (resized for overlay)
    base = (img_arr * 255).astype(np.uint8)
    overlay = base.copy()
    overlay[bin_mask == 1] = [255, 0, 0]  # red highlight

    blended = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)
    st.subheader("Overlay (Image + Predicted Mask)")
    st.image(blended, use_container_width=True)


