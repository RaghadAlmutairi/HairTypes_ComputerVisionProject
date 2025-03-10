import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model and cache it
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("advanced_densenet_model.h5")
        return model
    except Exception as e:
        st.error("Error loading model. Please check the file path and model format.")
        return None

# Define hair type class names (correct order based on training labels)
class_names = ["Straight", "Wavy", "Curly", "Dreadlocks", "Kinky"]

hair_tips = {
    "Curly": "Curly hair loves moisture! Use a leave-in conditioner and avoid brushing when dry.",
    "Dreadlocks": "Keep your dreadlocks clean by washing regularly with residue-free shampoo.",
    "Kinky": "Kinky hair thrives with protective styles like braids and twists. Moisturize daily!",
    "Straight": "Straight hair can get oily fast, so use dry shampoo between washes!",
    "Wavy": "Enhance your waves with a sea salt spray for that beachy look!"
}

fun_facts = {
    "Curly": "Did you know? Curly hair is naturally more prone to frizz because it absorbs more moisture!",
    "Dreadlocks": "Fun fact: Dreadlocks have been around for thousands of years in various cultures!",
    "Kinky": "Kinky hair has the most shrinkage but also the most volume when stretched!",
    "Straight": "Straight hair reflects light easily, which is why it looks shinier than other hair types!",
    "Wavy": "Wavy hair is the perfect middle ground between curly and straight, offering styling versatility!"
}

# Set up the Streamlit app layout
st.set_page_config(page_title="Hair Type Classification", page_icon="🌟", layout="wide")

# Sidebar for user inputs
st.sidebar.title("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main title
st.title("🌟 Hair Type Classification App 🌟")
st.write("Simply upload an image, and our AI model will classify the hair type and provide care tips!")

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image, dtype=np.float32) / 255.0  # Scale pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Load the model once
model = load_model()

# Layout structure
col1, col2 = st.columns([1, 2])

if uploaded_file is not None and model is not None:
    try:
        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        with col2:
            # Display results in a structured way
            st.subheader("Prediction Result")
            st.write(f"**Predicted Hair Type:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Display bar chart of all predictions
            prediction_data = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
            st.bar_chart(prediction_data)
        
        # Full width for extra information
        st.markdown("---")
        st.subheader("💡 Fun Fact")
        st.write(fun_facts.get(predicted_class, "No fun fact available."))
        
        st.subheader("✨ Hair Care Tip")
        st.write(hair_tips.get(predicted_class, "No hair care tip available."))
    
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

# Footer Information
st.markdown("---")
st.subheader("📌 About This Project")
st.write("This application uses an advanced deep learning model to classify hair types based on uploaded images.")
st.write("Developed by Team Visionaries, our goal is to combine AI and beauty to help people understand their hair type better.")
st.write("Using Streamlit and TensorFlow, we bring AI-powered hair analysis to everyone.")

st.markdown("Created with ❤ using Streamlit and TensorFlow 👀✨")

# Custom CSS for styling
st.markdown("""
    <style>
    .stImage {
        border: 2px solid #0073e6;
        border-radius: 10px;
    }
    .stSidebar > div {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    .stTitle {
        color: #0073e6;
    }
    .css-1g0s8g3 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
