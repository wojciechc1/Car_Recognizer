import streamlit as st
from pipelines.smart_analysis_pipeline import CarAnalysisPipeline
from pipelines.complex_analysis_pipeline import ComplexCarAnalysisPipeline

from PIL import Image
import tempfile
import os
import cv2
import numpy as np


def draw_boxes(image_bgr, detections, color):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image_bgr


smart_pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    "car": "./config/car_detector.pt",
    "color": "./config/color_classifier.pth"
    # "type": "...",
    # "model": "..."
})


complex_pipeline = ComplexCarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    "car": "./config/car_detector.pt",
    "color": "./config/color_classifier.pth"
    # "type": "...",
    # "model": "..."
})



st.title("🚗 Car Classifier Demo")

mode = st.radio("Select analysis mode::", ("Smart recognizer (1 photo)", "Complex recognition (3 photos)"))


if mode == "Smart recognizer (1 photo)":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Classification"):
            with st.spinner("🔍 Analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                    temp_path = temp.name
                    image.save(temp_path)

                results = smart_pipeline.run(temp_path)
                os.remove(temp_path)

            st.subheader("📊 Prediction Result:")
            st.json(results)


            # Jeśli są detekcje
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            if "logo" in results and results["logo"]:
                img_bgr = draw_boxes(img_bgr, results["logo"], (0, 255, 0))

            if "carbox" in results and results["carbox"]:
                img_bgr = draw_boxes(img_bgr, results["carbox"], (255, 0, 0))

            # Wyświetlenie w Streamlit
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="🟩 Detection", use_container_width=True)



if mode == "Complex recognition (3 photos)":

    col1, col2, col3 = st.columns(3)

    with col1:
        front_file = st.file_uploader("Front View", type=["jpg", "jpeg", "png"], key="front")
    with col2:
        side_file = st.file_uploader("Side View", type=["jpg", "jpeg", "png"], key="side")
    with col3:
        rear_file = st.file_uploader("Rear View", type=["jpg", "jpeg", "png"], key="rear")

    if st.button("Run Classification"):
        if not (front_file and side_file and rear_file):
            st.error("❌ Please upload all 3 images!")
        else:
            with st.spinner("🔍 Analyzing..."):
                temp_paths = {}

                # Zapisujemy 3 pliki tymczasowo
                for name, file in zip(
                        ["front", "side", "rear"],
                        [front_file, side_file, rear_file]
                ):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                        img = Image.open(file).convert("RGB")
                        img.save(temp.name)
                        temp_paths[name] = temp.name

                # Uruchomienie Twojego pipeline
                results = complex_pipeline.run(temp_paths)

                # Usuwamy pliki tymczasowe
                for path in temp_paths.values():
                    os.remove(path)

            st.subheader("📊 Prediction Result:")
            st.json(results)

            # Wizualizacja dla każdej perspektywy
            for view_name, file in zip(["front", "side", "rear"], [front_file, side_file, rear_file]):
                st.markdown(f"### {view_name.capitalize()} Detection")
                img = np.array(Image.open(file).convert("RGB"))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if view_name in results:
                    if "logo" in results[view_name] and results[view_name]["logo"]:
                        img_bgr = draw_boxes(img_bgr, results[view_name]["logo"], (0, 255, 0))
                    if "carbox" in results[view_name] and results[view_name]["carbox"]:
                        img_bgr = draw_boxes(img_bgr, results[view_name]["carbox"], (255, 0, 0))

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)