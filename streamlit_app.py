import streamlit as st
from pipelines.main_pipeline import CarAnalysisPipeline
from PIL import Image
import tempfile
import os


pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    # "type": "...",
    # "model": "..."
})



# Streamlit interfejs
st.title("Car Classifier Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Classification"):
        with st.spinner("Analyzing..."):
            # Zapisz obrazek tymczasowo na dysku

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp_path = temp.name

                image.save(temp_path)

            # Uruchom pipeline z tą ścieżką

            results = pipeline.run(temp_path)

            # Usuń plik tymczasowy

            os.remove(temp_path)

        st.subheader("🔍 Prediction Result:")

        st.json(results)