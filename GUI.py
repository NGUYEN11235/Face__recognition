import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFile
from PyQt5 import uic
import numpy as np
from MTCNN import MTCNN


class CameraThread(QThread):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, face_detection_model):
        super().__init__()
        self.capture = None
        self.face_detection_model = face_detection_model

    def run(self):
        self.capture = cv2.VideoCapture(0)
        while True:
            ret, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                # Run face detection model on frame
                bboxes, _ = self.face_detection_model.detect(frame)
                # Draw bounding boxes on frame
                if bboxes is not None:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Emit image data signal
                self.image_data.emit(frame)

    def stop(self):
        if self.capture is not None:
            self.capture.release()


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI file
        ui_file = QFile("form.ui")
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()

        self.camera_thread = None
        self.detection_model = MTCNN()

        # Connect signal/slot for buttons
        self.open_webcam_btn.clicked.connect(self.toggle_camera_thread)
        self.data_collect_rbtn.toggled.connect(self.set_image_view)
        self.run_rbtn.toggled.connect(self.set_image_view)

    def toggle_camera_thread(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread(self.detection_model)
            self.camera_thread.image_data.connect(self.update_image)
            self.camera_thread.start()
            self.open_webcam_btn.setText("Close Webcam")
        else:
            self.camera_thread.stop()
            self.camera_thread = None
            self.open_webcam_btn.setText("Open Webcam")

    def set_image_view(self):
        if self.data_collect_rbtn.isChecked():
            self.image_view = self.Image2_label
        else:
            self.image_view = self.Image1_label

    def update_image(self, np_image):
        # Resize and set QImage to QLabel
        q_image = self.convert_np_to_qimage(np_image)
        self.image_view.setPixmap(
            QPixmap.fromImage(q_image).scaled(self.image_view.width(), self.image_view.height(), Qt.KeepAspectRatio))


    def convert_np_to_qimage(self, np_image):
        h, w, ch = np_image.shape
        bytes_per_line = ch * w
        q_image = QImage(np_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())