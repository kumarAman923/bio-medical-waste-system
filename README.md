# Intelligent waste segregation system
This project demonstrates waste detection using a YOLOv8 (You Only Look Once) object detection model. It identifies recyclable, non-recyclable, and hazardous waste items in a webcam stream.

Our datasets used to train:
https://universe.roboflow.com/ai-project-i3wje/waste-detection-vqkjo/model/3

Colab:
https://colab.research.google.com/drive/1dHv5QUuz2NkkgzeKBoO4DLAhLg9mOrzv?usp=sharing




## Setup

```
**Install Dependencies:**
```bash
pip install -r requirements.txt
```
**Run the Application**
```bash
streamlit run app.py
```
Open your web browser and navigate to the provided URL (usually http://localhost:8501). You will see the bioMedical Waste System app.

## Project Structure

- `app.py`: Main application file containing Streamlit code.
- `helper.py`: Helper functions for waste detection using the YOLO model.
- `settings.py`: Configuration settings, including the path to the YOLO model and waste types.
- `train.py`: To train the model



## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLO Documentation](https://github.com/ultralytics/yolov5)

