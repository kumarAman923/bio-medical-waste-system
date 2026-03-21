from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading


def sleep_and_clear_success():
    time.sleep(3)
    st.session_state['infectious_placeholder'].empty()
    st.session_state['protective_placeholder'].empty()
    st.session_state['medical_placeholder'].empty()


def load_model(model_path):
    model = YOLO(model_path)
    return model


def classify_waste_type(detected_items):

    infectious_items = set(detected_items) & set(settings.INFECTIOUS)
    protective_items = set(detected_items) & set(settings.PROTECTIVE)
    medical_items = set(detected_items) & set(settings.MEDICAL_PLASTIC)

    return infectious_items, protective_items, medical_items


def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")


def _display_detected_frames(model, st_frame, image):

    image = cv2.resize(image, (640, int(640 * (9 / 16))))

    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    if 'infectious_placeholder' not in st.session_state:
        st.session_state['infectious_placeholder'] = st.sidebar.empty()

    if 'protective_placeholder' not in st.session_state:
        st.session_state['protective_placeholder'] = st.sidebar.empty()

    if 'medical_placeholder' not in st.session_state:
        st.session_state['medical_placeholder'] = st.sidebar.empty()

    if 'last_detection_time' not in st.session_state:
        st.session_state['last_detection_time'] = 0

    res = model.predict(image, conf=0.6)
    names = model.names
    detected_items = set()

    for result in res:

        new_classes = set([names[int(c)] for c in result.boxes.cls])

        if new_classes != st.session_state['unique_classes']:

            st.session_state['unique_classes'] = new_classes

            st.session_state['infectious_placeholder'].markdown('')
            st.session_state['protective_placeholder'].markdown('')
            st.session_state['medical_placeholder'].markdown('')

            detected_items.update(st.session_state['unique_classes'])

            infectious_items, protective_items, medical_items = classify_waste_type(detected_items)

            if infectious_items:
                detected_items_str = "\n- ".join(
                    remove_dash_from_class_name(item) for item in infectious_items
                )

                st.session_state['infectious_placeholder'].markdown(
                    f"<div style='background-color:#ff4d4d;padding:10px;border-radius:5px'>Infectious Waste:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )

            if protective_items:
                detected_items_str = "\n- ".join(
                    remove_dash_from_class_name(item) for item in protective_items
                )

                st.session_state['protective_placeholder'].markdown(
                    f"<div style='background-color:#4CAF50;padding:10px;border-radius:5px'>Protective Waste:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )

            if medical_items:
                detected_items_str = "\n- ".join(
                    remove_dash_from_class_name(item) for item in medical_items
                )

                st.session_state['medical_placeholder'].markdown(
                    f"<div style='background-color:#2196F3;padding:10px;border-radius:5px'>Medical Plastic Waste:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )

            threading.Thread(target=sleep_and_clear_success).start()

            st.session_state['last_detection_time'] = time.time()

    res_plotted = res[0].plot()

    st_frame.image(res_plotted, channels="BGR")


def play_webcam(model):

    source_webcam = settings.WEBCAM_PATH

    if st.button('Detect Biomedical Waste'):

        try:

            vid_cap = cv2.VideoCapture(source_webcam)

            st_frame = st.empty()

            while vid_cap.isOpened():

                success, image = vid_cap.read()

                if success:

                    _display_detected_frames(model, st_frame, image)

                else:
                    vid_cap.release()
                    break

        except Exception as e:

            st.sidebar.error("Error loading video: " + str(e))