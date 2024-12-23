import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('MobileNetV2_model.keras')
    return model

def preprocess_image(image, target_size):
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(model, image):
    processed_image = preprocess_image(image, target_size=(224, 224))
    predictions = model.predict(processed_image)
    return predictions

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main():
    # Load external CSS
    load_css("style.css")
    
    # Logo and title
    logo_base64 = load_image_as_base64('static/Kanindotama.jpg')

    st.markdown(f'<div class="center"><img src="data:image/jpeg;base64,{logo_base64}" width="100"></div>',unsafe_allow_html=True)
    st.markdown('<div class="center"><h1>Wood Scanner</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="center"><p>Arahkan kamera ke objek yang ingin dideteksi</p></div>', unsafe_allow_html=True)

    model = load_model()

    # Camera input
    enable_camera = st.checkbox("Aktifkan Kamera", value=True)

    if enable_camera:
        picture = st.camera_input("Ambil gambar")
    else:
        picture = None

    # File upload input
    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Pilih gambar", type=['jpg', 'jpeg', 'png'])

    image = None
    if picture is not None:
        image = Image.open(picture)
        st.image(image, caption='Gambar Tangkapan', use_container_width=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar Unggahan', use_container_width=True)

    # If an image is provided, make a prediction
    if image is not None:
        st.write("Memproses...")
        predictions = predict_image(model, image)

        # Display result as text
        if predictions == 1:
            st.write("Kayu Tidak Layak")
        else:
            st.write("Kayu Layak")


if __name__ == "__main__":
    main()