import streamlit as st
import tensorflow as tf
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("trained_model.h5")
    return model

# Load model immediately
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'trained_model.h5' is in the directory.")

# --- PREDICTION FUNCTION ---
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --- CLASS NAMES ---
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- CSS STYLING ---
# I removed the background-color rule so it works with your Dark Mode
st.markdown("""
    <style>
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white !important;
    }
    /* Custom Result Box Styling */
    .result-box {
        background-color: rgba(21, 87, 36, 0.2); 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #155724;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Plant Doctor")
    st.subheader("Navigation")
    app_mode = st.radio("Go to", ["Home", "Disease Recognition"])
    st.info("This application uses Deep Learning to identify plant diseases from leaf images.")

# --- MAIN PAGE LOGIC ---

if app_mode == "Home":
    st.title("🌿 Welcome to Plant Disease Recognition System")
    st.markdown("""
    ### Protect your crops with AI
    This intelligent system helps farmers and gardeners detect plant diseases early.
    
    **How it works:**
    1. Go to the **Disease Recognition** page.
    2. Upload an image of a plant leaf.
    3. Get an instant diagnosis.
    """)
    st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)

elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    col1, col2 = st.columns([1, 1])

    if test_image is not None:
        with col1:
            st.image(test_image, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.write("### Analysis")
            if st.button("Predict Disease"):
                with st.spinner("Analyzing image..."):
                    try:
                        result_index = model_prediction(test_image)
                        prediction = class_name[result_index]
                        
                        st.success("Analysis Complete!")
                        
                        # Formatting the result
                        plant_name = prediction.split('___')[0]
                        condition = prediction.split('___')[1].replace('_', ' ')
                        
                        # Display Custom HTML Result
                        st.markdown(f"""
                        <div class="result-box">
                            <h4 style="color: #4CAF50; margin:0;">Prediction Result:</h4>
                            <p style="font-size: 24px; font-weight: bold; color: #ffffff;">{plant_name} - {condition}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "healthy" in condition.lower():
                            st.write("✅ Your plant looks healthy!")
                        else:
                            st.warning("⚠️ Attention required. Consult an agricultural expert.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    else:
        st.info("Please upload an image to start.")