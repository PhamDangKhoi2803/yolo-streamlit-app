import io
from typing import Any
import cv2
import sys
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

class Inference:
    """
    A class to perform object detection, image classification, image segmentation, and pose estimation inference.
    """

    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")  # Ensure Streamlit is available

        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None
        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Set up the Streamlit web interface."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox("Video", ("webcam", "video"))
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Handle video file uploads through Streamlit."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # Use webcam index 0

    def configure(self):
        """Configure the model and load selected classes."""
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

    class VideoTransformer(VideoTransformerBase):
        def __init__(self, model, conf, iou, selected_ind, enable_trk):
            self.model = model
            self.conf = conf
            self.iou = iou
            self.selected_ind = selected_ind
            self.enable_trk = enable_trk

        def transform(self, frame):
            # Process frame with model
            img = frame.to_ndarray(format="bgr24")

            if self.enable_trk == "Yes":
                results = self.model.track(img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True)
            else:
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)

            annotated_frame = results[0].plot()

            return annotated_frame

    def inference(self):
        """Perform real-time object detection on video or webcam feed."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()
    
        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")
    
            # Kiểm tra nếu nguồn là webcam
            if self.source == "webcam":
                webrtc_streamer(
                    key="object-detection",
                    video_transformer_factory=lambda: self.VideoTransformer(self.model, self.conf, self.iou, self.selected_ind, self.enable_trk),
                    video_source=0,  # Truyền chỉ số webcam, không phải file
                    sendback_audio=False,
                )
            elif self.source == "video" and self.vid_file_name != "":
                webrtc_streamer(
                    key="object-detection",
                    video_transformer_factory=lambda: self.VideoTransformer(self.model, self.conf, self.iou, self.selected_ind, self.enable_trk),
                    video_source=self.vid_file_name,  # Truyền video file nếu chọn nguồn là video
                    sendback_audio=False,
                )
    
            # Handle stopping of inference
            if stop_button:
                self.st.stop()


if __name__ == "__main__":
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None
    Inference(model=model).inference()
