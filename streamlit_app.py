import streamlit as st

# Streamlit interfejs
st.title("Car Classifier Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Classification"):
        result = classify_image(image)
        st.subheader("Prediction Result (JSON):")
        st.json(result)