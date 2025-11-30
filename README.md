# 🖼️ Image Segmentation using U-Net

This project performs **image segmentation** using a **U-Net** based convolutional neural network.  
Given an input image, the model predicts a **segmentation mask** that highlights the region of interest .

---

## 📌 What is Image Segmentation?

Image segmentation is the task of **classifying each pixel** in an image.  
Unlike image classification (one label for the whole image), segmentation tells us **exactly where** an object is.



## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Matplotlib  
- Streamlit for simple UI

---

## 📂 Project Structure


project/
│── data/
│   ├── images/         # Input images
│   └── masks/          # Ground truth masks
│
│── src/
│   ├── unet_model.py   # U-Net architecture
│   └── code.ipynb        # VS code
│
│── app.py              # Streamlit app
│── README.md
