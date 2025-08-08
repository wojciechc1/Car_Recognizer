import streamlit as st
from pipelines.main_pipeline import CarAnalysisPipeline
from PIL import Image
import numpy as np
import cv2
import tempfile

pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    # "type": "...",
    # "model": "..."
})

# Funkcja do rysowania ramki
def draw_logo_boxes(image: np.ndarray, logos: list) -> np.ndarray:
    for logo in logos:
        label = logo["label"]
        confidence = logo["confidence"]
        bbox = logo["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image


# Streamlit interfejs
st.title("Car Classifier Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("Run Classification"):
        with st.spinner("🔍 Analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp_path = temp.name
                image.save(temp_path)

            results = pipeline.run(temp_path)
            os.remove(temp_path)

        st.subheader("📊 Prediction Result:")
        st.json(results)

        # Jeśli jest detekcja logo, narysuj
        if "logo" in results and results["logo"]:
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_with_boxes = draw_logo_boxes(img_bgr, results["logo"])
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="🟩 Logo Detection", use_column_width=True)