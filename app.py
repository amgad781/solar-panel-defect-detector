import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Solar Panel Defect Classifier", page_icon="☀️")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("☀️ Solar Panel Defect & Dust Classifier")
st.write("Upload a photo of a solar panel surface to detect dust or damage.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        results = model(image, verbose=False)
        probs = results[0].probs
        class_names = results[0].names

    top_class = class_names[probs.top1]
    confidence = probs.top1conf.item()

    st.subheader(f"Prediction: **{top_class}**")
    st.write(f"Confidence: {confidence:.1%}")

    st.write("---")
    st.write("All class probabilities:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {float(probs.data[i]):.1%}")
        st.progress(float(probs.data[i]))