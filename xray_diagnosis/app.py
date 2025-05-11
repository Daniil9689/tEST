import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"

st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Добавление фона через HTML
st.markdown("""
    <style>
        .background {
            background-image: url('https://cdn.pixabay.com/photo/2017/08/30/03/26/space-2695560_960_720.jpg');
            background-size: cover;
            background-position: center;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        .content {
            position: relative;
            z-index: 1;
            color: white;
        }
        .label {
            font-weight: 600;
            font-size: 16px;
        }
        .bar-container {
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            height: 20px;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        .bar {
            height: 100%;
            border-radius: 5px;
        }
        h1, h2, h3 {
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        }
    </style>
""", unsafe_allow_html=True)

# HTML для фонового изображения
st.markdown('<div class="background"></div>', unsafe_allow_html=True)

st.title("🩺 Классификация заболеваний по рентгену")
st.markdown("Загрузите изображение грудной клетки, и модель покажет вероятность каждого заболевания.")

uploaded_file = st.file_uploader("📤 Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Загруженное изображение", use_column_width=True)
    st.markdown("### ⏳ Анализируем изображение...")

    model = load_model()
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0]

    st.markdown("## 🧾 Результаты классификации")

    for name, prob in zip(CLASS_NAMES, prediction):
        percent = prob * 100
        color = "#d9534f" if percent > 60 else "#f0ad4e" if percent > 30 else "#5bc0de"
        st.markdown(
            f"""
            <div class="label">{name}: <span style='color:{color}'>{percent:.2f}%</span></div>
            <div class="bar-container">
                <div class="bar" style="width:{percent}%; background-color:{color}"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
