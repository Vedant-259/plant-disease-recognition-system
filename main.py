import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np



st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 4em;
        padding-bottom: 20px;
        color: #008000;
    }
    .image-container {
        display: flex;
        justify-content: center;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    .about-text {
        color: #008000;
        margin: 0px 0px;
    }
    .sidebar .sidebar-content .sidebar-section .sidebar-container .sidebar-components .sidebar {
        font-size: 5.2rem;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

def model_predictions(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_plant_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)
    input_arr= np.array([input_arr]) # convert single image to a batch
    predictions=model.predict(input_arr)
    return np.argmax(predictions) 


st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Choose One", ["Home", "About", "Disease Recognition"])

def centered_header(header_text):
    st.markdown(f"<h1 class='main-header'>{header_text}</h1>", unsafe_allow_html=True)


if app_mode == "Home":
    centered_header("Plant Disease Recognition System")
    image_path = "images.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System, an innovative tool designed to help farmers and agricultural professionals quickly and accurately identify plant diseases. Using state-of-the-art machine learning algorithms and image processing techniques, our system analyzes images of plants to detect a variety of diseases, enabling early intervention and effective treatment.

        This system not only helps in maintaining the health of your crops but also optimizes yield and quality. By reducing the dependency on chemical treatments through precise diagnosis, it promotes sustainable farming practices. Join us in revolutionizing agriculture with cutting-edge technology, ensuring a healthier and more productive future for our farms.

        **Features:**
        - **Accurate Disease Detection:** Utilizes advanced machine learning models trained on a vast dataset of plant images.
        - **User-Friendly Interface:** Easy to upload images and get instant predictions.
        - **Detailed Results:** Provides comprehensive information about the identified disease and potential treatment options.
        - **Sustainable Farming:** Aims to reduce chemical usage by enabling targeted treatments.

        Explore our system to see how it can benefit your agricultural practices and help maintain the health and productivity of your crops.
    """, unsafe_allow_html=True)


elif app_mode == "About":
    centered_header("About")
    st.markdown("""
        Plant disease recognition is a crucial aspect of modern agriculture, leveraging advanced technologies such as machine learning and image processing to identify and diagnose diseases in plants. This innovative approach enables farmers and agricultural professionals to detect diseases early, ensuring timely intervention and preventing the spread of infections. By utilizing high-resolution images and sophisticated algorithms, plant disease recognition systems can accurately classify various diseases based on visual symptoms, such as leaf spots, discoloration, and mold growth. This not only improves crop yield and quality but also reduces the reliance on chemical treatments, promoting more sustainable farming practices. The integration of plant disease recognition technology into agricultural workflows represents a significant step forward in achieving efficient and resilient food production systems.
        
        **Testing Dataset**
        1. **Train:** 70,295 images
        2. **Test:** 30 images
        3. **Valid:** 17,572 images
    """, unsafe_allow_html=True)

elif(app_mode =="Disease Recognition"):
    centered_header("Disease Recognition")
    test_image=st.file_uploader("Choose an image:")
    if(st.button("show_image")):
        st.image(test_image,width=4,use_column_width=True)

    if(st.button("Predict")):
        st.snow()   
        st.write("Our Prdeiction")
        result_index = model_predictions(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        st.success("Model is predicting it's a {}".format(class_name[result_index]))