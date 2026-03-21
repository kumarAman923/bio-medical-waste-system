# from pathlib import Path
# import streamlit as st
# import helper
# import settings

# st.set_page_config(
#     page_title="Biomedical Waste Detection",
# )

# st.sidebar.title("Detect Console")

# model_path = Path(settings.DETECTION_MODEL)

# st.title("Biomedical Waste Detection System")

# st.write(
# "Start detecting biomedical waste objects in the webcam stream by clicking the button below."
# )

# st.markdown(
# """
# <style>

# .stInfectious {
#     background-color: rgba(255,0,0,0.3);
#     padding: 1rem 0.75rem;
#     border-radius: 0.5rem;
#     font-size:18px !important;
# }

# .stProtective {
#     background-color: rgba(0,255,0,0.3);
#     padding: 1rem 0.75rem;
#     border-radius: 0.5rem;
#     font-size:18px !important;
# }

# .stMedical {
#     background-color: rgba(0,0,255,0.3);
#     padding: 1rem 0.75rem;
#     border-radius: 0.5rem;
#     font-size:18px !important;
# }

# </style>
# """,
# unsafe_allow_html=True
# )

# try:
#     model = helper.load_model(model_path)
# except Exception as ex:
#     st.error(f"Unable to load model. Check the specified path: {model_path}")
#     st.error(ex)

# helper.play_webcam(model)

# st.sidebar.markdown("Biomedical Waste Detection Demo", unsafe_allow_html=True)





import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load trained classification model
model = YOLO("best.pt")

st.title("Biomedical Waste Classification System")

option = st.radio("Select Input Method", ["Upload Image", "Camera"])

# -------------------
# Image Upload
# -------------------
if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload waste image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image)

        results = model.predict(image)

        class_id = results[0].probs.top1
        class_name = model.names[class_id]

        st.success(f"Predicted Waste: {class_name}")

# -------------------
# Camera Detection
# -------------------
if option == "Camera":

    camera_image = st.camera_input("Take Photo")

    if camera_image is not None:

        bytes_data = camera_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(image)

        class_id = results[0].probs.top1
        class_name = model.names[class_id]

        st.image(image, channels="BGR")
        st.success(f"Predicted Waste: {class_name}")