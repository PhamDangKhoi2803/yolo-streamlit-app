# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any

import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class VideoProcessor(VideoProcessorBase):
    """Lớp xử lý video cho streamlit-webrtc để xử lý khung hình webcam."""
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.iou = 0.45
        self.selected_ind = []
        self.enable_trk = "No"

    def set_model(self, model, conf, iou, selected_ind, enable_trk):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.selected_ind = selected_ind
        self.enable_trk = enable_trk

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Chuyển frame sang định dạng BGR cho OpenCV
        if self.model:
            if self.enable_trk == "Yes":
                results = self.model.track(
                    img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                )
            else:
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        return frame


class Inference:
    """
    Lớp để thực hiện suy luận phát hiện đối tượng, phân loại hình ảnh, phân đoạn hình ảnh và ước lượng tư thế.

    Lớp này cung cấp các chức năng để tải mô hình, cấu hình cài đặt, tải lên tệp video và thực hiện suy luận
    thời gian thực bằng Streamlit và các mô hình Ultralytics YOLO.

    Thuộc tính:
        st (module): Module Streamlit để tạo giao diện người dùng.
        temp_dict (dict): Từ điển tạm thời để lưu đường dẫn mô hình và các cấu hình khác.
        model_path (str): Đường dẫn đến mô hình đã tải.
        model (YOLO): Thể hiện của mô hình YOLO.
        source (str): Nguồn video được chọn (webcam hoặc tệp video).
        enable_trk (str): Tùy chọn bật theo dõi ("Có" hoặc "Không").
        conf (float): Ngưỡng độ tin cậy cho phát hiện.
        iou (float): Ngưỡng IoU cho non-maximum suppression.
        org_frame (Any): Vùng chứa cho khung hình gốc được hiển thị.
        ann_frame (Any): Vùng chứa cho khung hình đã chú thích được hiển thị.
        vid_file_name (str | int): Tên tệp video đã tải lên hoặc chỉ số webcam.
        selected_ind (List[int]): Danh sách các chỉ số lớp được chọn để phát hiện.
    """

    def __init__(self, **kwargs: Any):
        """
        Khởi tạo lớp Inference, kiểm tra yêu cầu Streamlit và thiết lập đường dẫn mô hình.

        Args:
            **kwargs (Any): Các đối số từ khóa bổ sung cho cấu hình mô hình.
        """
        check_requirements("streamlit>=1.29.0")  # Kiểm tra yêu cầu Streamlit
        import streamlit as st

        self.st = st  # Tham chiếu đến module Streamlit
        self.source = None  # Lựa chọn nguồn video (webcam hoặc tệp video)
        self.enable_trk = False  # Cờ để bật/tắt theo dõi đối tượng
        self.conf = 0.25  # Ngưỡng độ tin cậy cho phát hiện
        self.iou = 0.45  # Ngưỡng IoU cho non-maximum suppression
        self.org_frame = None  # Vùng chứa cho khung hình gốc
        self.ann_frame = None  # Vùng chứa cho khung hình chú thích
        self.vid_file_name = None  # Tên tệp video hoặc chỉ số webcam
        self.selected_ind = []  # Danh sách chỉ số lớp được chọn
        self.model = None  # Thể hiện mô hình YOLO

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # Đường dẫn tệp mô hình
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Thiết lập giao diện web Streamlit với các yếu tố HTML tùy chỉnh."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Ẩn menu chính

        # Tiêu đề chính của ứng dụng Streamlit
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ứng dụng Streamlit Ultralytics YOLO</h1></div>"""

        # Phụ đề của ứng dụng Streamlit
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Trải nghiệm phát hiện đối tượng thời gian thực trên webcam với sức mạnh 
        của Ultralytics YOLO! 🚀</h4></div>"""

        # Thiết lập cấu hình trang HTML và thêm HTML tùy chỉnh
        self.st.set_page_config(page_title="Ứng dụng Streamlit Ultralytics", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Cấu hình thanh bên Streamlit cho các cài đặt mô hình và suy luận."""
        with self.st.sidebar:  # Thêm logo Ultralytics
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("Cấu hình người dùng")  # Thêm các yếu tố vào menu cài đặt dọc
        self.source = self.st.sidebar.selectbox(
            "Nguồn video",
            ("webcam", "video"),
        )  # Thêm dropdown chọn nguồn
        self.enable_trk = self.st.sidebar.radio("Bật theo dõi", ("Có", "Không"))  # Bật theo dõi đối tượng
        self.conf = float(
            self.st.sidebar.slider("Ngưỡng độ tin cậy", 0.0, 1.0, self.conf, 0.01)
        )  # Thanh trượt cho độ tin cậy
        self.iou = float(self.st.sidebar.slider("Ngưỡng IoU", 0.0, 1.0, self.iou, 0.01))  # Thanh trượt cho ngưỡng NMS

        col1, col2 = self.st.columns(2)  # Tạo hai cột để hiển thị khung hình
        self.org_frame = col1.empty()  # Vùng chứa cho khung hình gốc
        self.ann_frame = col2.empty()  # Vùng chứa cho khung hình chú thích

    def source_upload(self):
        """Xử lý tải lên tệp video hoặc chọn webcam qua giao diện Streamlit."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Tải lên tệp video", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # Đối tượng BytesIO
                with open("ultralytics.mp4", "wb") as out:  # Mở tệp tạm dưới dạng bytes
                    out.write(g.read())  # Đọc bytes vào tệp
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = "webrtc"  # Đánh dấu nguồn là webcam qua webrtc

    def configure(self):
        """Cấu hình mô hình và tải các lớp được chọn để suy luận."""
        # Thêm menu dropdown để chọn mô hình
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:  # Nếu người dùng cung cấp mô hình tùy chỉnh, thêm mô hình vào danh sách
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Mô hình", available_models)

        with self.st.spinner("Đang tải mô hình..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")  # Tải mô hình YOLO
            class_names = list(self.model.names.values())  # Chuyển từ điển thành danh sách tên lớp
        self.st.success("Tải mô hình thành công!")

        # Hộp chọn nhiều lớp với tên lớp và lấy chỉ số của các lớp được chọn
        selected_classes = self.st.sidebar.multiselect("Lớp", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # Đảm bảo selected_ind là danh sách
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Thực hiện suy luận phát hiện đối tượng thời gian thực trên video hoặc webcam."""
        self.web_ui()  # Khởi tạo giao diện web
        self.sidebar()  # Tạo thanh bên
        self.source_upload()  # Tải lên nguồn video
        self.configure()  # Cấu hình ứng dụng

        if self.st.sidebar.button("Bắt đầu"):
            stop_button = self.st.button("Dừng")
            if self.vid_file_name == "webrtc":
                # Sử dụng streamlit-webrtc cho webcam
                webrtc_ctx = webrtc_streamer(
                    key="example",
                    video_processor_factory=VideoProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                )
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.set_model(
                        self.model, self.conf, self.iou, self.selected_ind, self.enable_trk
                    )
            else:
                # Xử lý tệp video với OpenCV
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    self.st.error("Không thể mở nguồn video.")
                    return

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("Không thể đọc khung hình từ nguồn video.")
                        break

                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                    annotated_frame = results[0].plot()

                    if stop_button:
                        cap.release()
                        self.st.stop()

                    self.org_frame.image(frame, channels="BGR")
                    self.ann_frame.image(annotated_frame, channels="BGR")

                cap.release()  # Giải phóng tài nguyên video


if __name__ == "__main__":
    import sys  # Nhập module sys để truy cập đối số dòng lệnh

    # Kiểm tra nếu tên mô hình được cung cấp qua đối số dòng lệnh
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Gán đối số đầu tiên làm tên mô hình nếu có
    # Tạo thể hiện của lớp Inference và chạy suy luận
    Inference(model=model).inference()
