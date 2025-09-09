import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QGroupBox, QGridLayout, QMessageBox, QDialog, QRubberBand,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QPalette
from PIL import Image
import traceback

DISPLAY_W, DISPLAY_H = 400, 160  # Area for result display (smaller size)

def cvimg_to_qpixmap(img, w=DISPLAY_W, h=DISPLAY_H):
    """Safely convert CV2 image to QPixmap with proper error handling"""
    try:
        if img is None:
            return QPixmap()
        
        # Ensure image is in correct format
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        
        # Convert to PIL for reliable resizing
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Convert to QImage
        data = pil_img.tobytes()
        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"Error converting image to pixmap: {e}")
        return QPixmap()

def safe_image_to_qpixmap(cv_img, max_width=800, max_height=600):
    """Safely convert CV2 image to QPixmap for display"""
    try:
        if cv_img is None:
            return QPixmap(), (0, 0)
        
        h, w = cv_img.shape[:2]
        original_size = (h, w)
        
        # Calculate scaling to fit within max dimensions
        scale_w = max_width / w if w > max_width else 1.0
        scale_h = max_height / h if h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert color space safely
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif len(cv_img.shape) == 2:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv_img
        
        # Create QImage
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(qimg), original_size
    except Exception as e:
        print(f"Error in safe_image_to_qpixmap: {e}")
        traceback.print_exc()
        return QPixmap(), (0, 0)

def findcontours(image1, image2, threshvalue, linestep):
    try:
        # Ensure images are valid
        if image1 is None or image2 is None:
            raise ValueError("One or both images are None")
        
        # Resize image2 to match image1 dimensions
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        original = image1.copy()
        
        # Convert to grayscale safely
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
        # Process difference
        difference = cv2.absdiff(gray_image2, gray_image1)
        blur = cv2.GaussianBlur(difference, (19, 19), cv2.BORDER_DEFAULT)
        ret, thresh = cv2.threshold(blur, threshvalue, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchies = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Draw contours
        contoured = original.copy()
        cv2.drawContours(contoured, contours, -1, (0, 0, 0), 5)
        
        # Create mask
        mask = np.zeros_like(original)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # Apply mask
        result = cv2.bitwise_and(original, mask)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Generate line coordinates
        line_coordinates = []
        for x in range(result_gray.shape[1]):
            col_pixels = result_gray[:, x]
            if cv2.countNonZero(col_pixels) != 0:
                y_coordinates = np.where(col_pixels > 0)[0]
                line_coordinates.append([(x, y) for y in y_coordinates])
        
        # Create lines
        line_coordinates_np = [np.array(line, dtype=np.int32) for line in line_coordinates]
        lines = []
        for i in range(0, len(line_coordinates_np), max(1, linestep)):
            if i < len(line_coordinates_np):
                lines.append(line_coordinates_np[i])
        
        # Draw lines
        lined = contoured.copy()
        if lines:
            cv2.polylines(lined, lines, isClosed=False, color=(255, 255, 255), thickness=2)
        
        # Combine results
        combined = np.hstack([original, contoured, lined])
        return combined, contours
    except Exception as e:
        print(f"Error in findcontours: {e}")
        traceback.print_exc()
        raise

def save_selected_contours(image, contours, filepath):
    try:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, mask)
        
        # Make transparent PNG
        b, g, r = cv2.split(result)
        alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        rgba = cv2.merge([b, g, r, alpha])
        cv2.imwrite(filepath, rgba)
    except Exception as e:
        print(f"Error saving contours: {e}")
        raise

class CropDialog(QDialog):
    def __init__(self, cv_image):
        super().__init__()
        self.setWindowTitle("Step 2: Select Control Region (Crop)")
        self.cv_image = cv_image.copy()  # Store original CV2 image
        self.crop_rect = None
        self.is_cropping = False
        
        try:
            # Get original dimensions
            self.original_h, self.original_w = cv_image.shape[:2]
            
            # Convert to display pixmap with scaling
            self.display_pixmap, _ = safe_image_to_qpixmap(cv_image, 800, 600)
            
            if self.display_pixmap.isNull():
                raise ValueError("Failed to create display pixmap")
            
            # Calculate scale factors
            self.scale_x = self.original_w / self.display_pixmap.width()
            self.scale_y = self.original_h / self.display_pixmap.height()
            
            # Setup UI
            self.setup_ui()
            
        except Exception as e:
            print(f"Error initializing CropDialog: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to initialize crop dialog: {str(e)}")
            self.reject()

    def setup_ui(self):
        try:
            # Create scroll area for large images
            self.scroll_area = QScrollArea()
            self.image_label = QLabel()
            self.image_label.setPixmap(self.display_pixmap)
            self.image_label.setScaledContents(False)
            
            # Setup rubber band
            self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.image_label)
            self.origin = QPoint()
            
            # Connect mouse events
            self.image_label.mousePressEvent = self.mouse_press_event
            self.image_label.mouseMoveEvent = self.mouse_move_event
            self.image_label.mouseReleaseEvent = self.mouse_release_event
            
            self.scroll_area.setWidget(self.image_label)
            self.scroll_area.setWidgetResizable(True)
            
            # Layout
            layout = QVBoxLayout()
            
            # Instructions
            instruction_label = QLabel("Drag to select the control region from the injury image.")
            instruction_label.setWordWrap(True)
            layout.addWidget(instruction_label)
            
            # Image info
            info_label = QLabel(f"Original: {self.original_w}×{self.original_h}, Display: {self.display_pixmap.width()}×{self.display_pixmap.height()}")
            layout.addWidget(info_label)
            
            layout.addWidget(self.scroll_area)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            self.setLayout(layout)
            
            # Set dialog size
            dialog_width = min(900, self.display_pixmap.width() + 50)
            dialog_height = min(700, self.display_pixmap.height() + 120)
            self.resize(dialog_width, dialog_height)
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            traceback.print_exc()
            raise

    def mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self.is_cropping = True

    def mouse_move_event(self, event):
        if self.is_cropping:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouse_release_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_cropping:
            self.crop_rect = self.rubber_band.geometry()
            self.is_cropping = False

    def get_crop_coordinates(self):
        """Return crop coordinates in original image space"""
        if self.crop_rect is None or self.crop_rect.isEmpty():
            return None
            
        # Convert display coordinates to original image coordinates
        x = max(0, int(self.crop_rect.x() * self.scale_x))
        y = max(0, int(self.crop_rect.y() * self.scale_y))
        w = int(self.crop_rect.width() * self.scale_x)
        h = int(self.crop_rect.height() * self.scale_y)
        
        # Ensure coordinates are within bounds
        x = min(x, self.original_w - 1)
        y = min(y, self.original_h - 1)
        w = min(w, self.original_w - x)
        h = min(h, self.original_h - y)
        
        return (x, y, w, h)

class InjuryDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboDoc v1 - Injury Segmentation")
        self.injury_img = None
        self.cropped_control = None
        self.result_img = None
        self.contours = None
        self.threshvalue = 50
        self.linestep = 10
        
        self.setup_ui()
        self.apply_professional_styling()

    def create_header(self):
        """Create professional header with logo and title"""
        header_frame = QFrame()

        
        header_layout = QVBoxLayout()
        header_layout.setSpacing(10)
        
        # Logo
        logo_label = QLabel()
        pixmap = QPixmap("RoboDoc 1/logo.png")
        logo_pixmap = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title_label = QLabel("RoboDoc v1")
        title_font = QFont()
        title_font.setPointSize(30)
        title_font.setBold(True)
        title_font.setFamily("Arial")
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin: 0px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the title

        subtitle_label = QLabel("Injury Segmentation & Analysis")
        subtitle_font = QFont()
        subtitle_font.setPointSize(16)
        subtitle_font.setFamily("Arial")
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: #7f8c8d; margin: 0px;")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the subtitle
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()
        
        header_layout.addWidget(logo_label)
        header_layout.addLayout(title_layout, 1)
        
        header_frame.setLayout(header_layout)
        return header_frame

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Add professional header
        header = self.create_header()
        main_layout.addWidget(header)

        # Step 1: Upload injury image
        step1_group = QGroupBox("Step 1: Upload Injury Image")
        step1_group.setStyleSheet(self.get_groupbox_style())
        step1_layout = QHBoxLayout()
        step1_layout.setSpacing(15)
        
        self.injury_label = QLabel("No injury image selected")
        self.injury_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.upload_injury_btn = QPushButton("📁 Upload Injury Image")
        self.upload_injury_btn.setStyleSheet(self.get_button_style())
        
        step1_layout.addWidget(self.injury_label, 1)
        step1_layout.addWidget(self.upload_injury_btn)
        step1_group.setLayout(step1_layout)
        main_layout.addWidget(step1_group)

        # Step 2: Crop control region
        step2_group = QGroupBox("Step 2: Select Control Region")
        step2_group.setStyleSheet(self.get_groupbox_style())
        step2_layout = QHBoxLayout()
        step2_layout.setSpacing(15)
        
        step2_info = QLabel("Crop a healthy region from the injury image for comparison.")
        step2_info.setWordWrap(True)
        step2_info.setStyleSheet("color: #34495e;")
        
        self.crop_btn = QPushButton("✂️ Select Control Region")
        self.crop_btn.setEnabled(False)
        self.crop_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        step2_layout.addWidget(step2_info, 1)
        step2_layout.addWidget(self.crop_btn)
        step2_group.setLayout(step2_layout)
        main_layout.addWidget(step2_group)

        # Step 3: Parameters
        step3_group = QGroupBox("Step 3: Adjust Detection Parameters")
        step3_group.setStyleSheet(self.get_groupbox_style())
        param_layout = QGridLayout()
        param_layout.setSpacing(15)
        
        # Threshold parameter
        thresh_label = QLabel("Sensitivity Threshold:")
        thresh_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setMinimum(3)
        self.thresh_slider.setMaximum(190)
        self.thresh_slider.setValue(self.threshvalue)
        self.thresh_slider.valueChanged.connect(self.update_thresh)
        self.thresh_slider.setStyleSheet(self.get_slider_style())
        
        self.thresh_value_label = QLabel(str(self.threshvalue))
        self.thresh_value_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 30px;")
        
        # Line step parameter
        line_label = QLabel("Line Density:")
        line_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.line_slider = QSlider(Qt.Orientation.Horizontal)
        self.line_slider.setMinimum(1)
        self.line_slider.setMaximum(50)
        self.line_slider.setValue(self.linestep)
        self.line_slider.valueChanged.connect(self.update_line)
        self.line_slider.setStyleSheet(self.get_slider_style())
        
        self.line_value_label = QLabel(str(self.linestep))
        self.line_value_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 30px;")
        
        param_layout.addWidget(thresh_label, 0, 0)
        param_layout.addWidget(self.thresh_slider, 0, 1)
        param_layout.addWidget(self.thresh_value_label, 0, 2)
        param_layout.addWidget(line_label, 1, 0)
        param_layout.addWidget(self.line_slider, 1, 1)
        param_layout.addWidget(self.line_value_label, 1, 2)
        
        step3_group.setLayout(param_layout)
        main_layout.addWidget(step3_group)

        # Step 4: Results
        step4_group = QGroupBox("Step 4: Processing & Export")
        step4_group.setStyleSheet(self.get_groupbox_style())
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.result_btn = QPushButton("🔬 Analyze Image")
        self.result_btn.setEnabled(False)
        self.result_btn.setStyleSheet(self.get_button_style(disabled=True, primary=True))
        
        self.save_btn = QPushButton("💾 Save Analysis")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        self.save_contour_btn = QPushButton("📄 Export Contours")
        self.save_contour_btn.setEnabled(False)
        self.save_contour_btn.setStyleSheet(self.get_button_style(disabled=True))
        
        btn_layout.addWidget(self.result_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.save_contour_btn)
        btn_layout.addStretch()
        step4_group.setLayout(btn_layout)
        main_layout.addWidget(step4_group)

        # Result display
        result_section = QGroupBox("Analysis Results")
        result_section.setStyleSheet(self.get_groupbox_style())
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setFixedSize(DISPLAY_W, DISPLAY_H)  # Use new smaller size
        self.result_label.setStyleSheet("""
            border: 2px dashed #bdc3c7; 
            background-color: #f8f9fa;
            border-radius: 8px;
        """)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setText("Analysis results will appear here")
        
        result_layout.addWidget(self.result_label, 0, Qt.AlignmentFlag.AlignCenter)
        result_section.setLayout(result_layout)
        main_layout.addWidget(result_section)

        main_layout.addStretch()
        self.setLayout(main_layout)
        
        # Connect signals
        self.upload_injury_btn.clicked.connect(self.upload_injury)
        self.crop_btn.clicked.connect(self.crop_control)
        self.result_btn.clicked.connect(self.show_result)
        self.save_btn.clicked.connect(self.save_result)
        self.save_contour_btn.clicked.connect(self.save_selected_contours)

    def get_groupbox_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: #ffffff;
                color: #2c3e50;
            }
        """

    def get_button_style(self, disabled=False, primary=False):
        if disabled:
            return """
                QPushButton {
                    background-color: #ecf0f1;
                    border: 1px solid #bdc3c7;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 12px;
                    color: #95a5a6;
                    min-width: 120px;
                }
            """
        elif primary:
            return """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    border: 1px solid #2980b9;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    color: white;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #5dade2, stop: 1 #3498db);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2980b9, stop: 1 #21618c);
                }
            """
        else:
            return """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #ffffff, stop: 1 #f8f9fa);
                    border: 1px solid #bdc3c7;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 12px;
                    color: #2c3e50;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #f8f9fa, stop: 1 #e9ecef);
                    border: 1px solid #95a5a6;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #e9ecef, stop: 1 #dee2e6);
                }
            """

    def get_slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 6px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #ecf0f1, stop: 1 #d5dbdb);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #3498db, stop: 1 #2980b9);
                border: 1px solid #2980b9;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #5dade2, stop: 1 #3498db);
            }
        """

    def apply_professional_styling(self):
        """Apply professional styling to the main window"""
        self.setStyleSheet("""
            QWidget {
                font-family: Arial, sans-serif;
                font-size: 11px;
                background-color: #f5f6fa;
            }
            QLabel {
                color: #2c3e50;
            }
        """)
        self.setMinimumSize(800, 900)
        self.resize(900, 1000)

    def upload_injury(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Injury Image", "", 
                "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
            )
            
            if not path:
                return
                
            # Load image
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Could not load image. Please check the file format.")
                return
            
            # Validate image
            if len(img.shape) != 3 or img.shape[2] != 3:
                QMessageBox.warning(self, "Error", "Please select a color image (BGR/RGB format).")
                return
                
            self.injury_img = img
            filename = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
            self.injury_label.setText(f"✅ {filename} ({img.shape[1]}×{img.shape[0]})")
            self.injury_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            # Enable crop button and update its style
            self.crop_btn.setEnabled(True)
            self.crop_btn.setStyleSheet(self.get_button_style())
            
            # Reset state
            self.cropped_control = None
            self.result_btn.setEnabled(False)
            self.result_btn.setStyleSheet(self.get_button_style(disabled=True, primary=True))
            self.save_btn.setEnabled(False)
            self.save_btn.setStyleSheet(self.get_button_style(disabled=True))
            self.save_contour_btn.setEnabled(False)
            self.save_contour_btn.setStyleSheet(self.get_button_style(disabled=True))
            self.result_label.clear()
            self.result_label.setText("📁 Image loaded successfully!\nProceed to select control region.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            print(f"Upload error: {e}")
            traceback.print_exc()

    def crop_control(self):
        if self.injury_img is None:
            return
            
        try:
            dialog = CropDialog(self.injury_img)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                coords = dialog.get_crop_coordinates()
                
                if coords is None:
                    QMessageBox.warning(self, "No Selection", "Please select a region to crop.")
                    return
                
                x, y, w, h = coords
                
                if w <= 10 or h <= 10:
                    QMessageBox.warning(self, "Invalid Selection", "Selected region is too small.")
                    return
                
                # Crop the image
                self.cropped_control = self.injury_img[y:y+h, x:x+w].copy()
                
                if self.cropped_control.size == 0:
                    QMessageBox.warning(self, "Crop Error", "Failed to crop image.")
                    return
                
                # Enable result button and update its style
                self.result_btn.setEnabled(True)
                self.result_btn.setStyleSheet(self.get_button_style(primary=True))
                
                self.result_label.setText(f"✂️ Control region selected: {w}×{h} pixels\nReady for analysis - click 'Analyze Image'")
                QMessageBox.information(self, "Success", 
                                      f"Control region selected: {w}×{h} pixels\nYou can now proceed with the analysis.")
                
        except Exception as e:
            QMessageBox.critical(self, "Crop Error", f"Failed to crop image: {str(e)}")
            print(f"Crop error: {e}")
            traceback.print_exc()

    def update_thresh(self, val):
        self.threshvalue = val
        self.thresh_value_label.setText(str(val))

    def update_line(self, val):
        self.linestep = val
        self.line_value_label.setText(str(val))

    def show_result(self):
        if self.injury_img is None or self.cropped_control is None:
            return
            
        try:
            self.result_label.setText("🔬 Processing analysis...\nPlease wait...")
            QApplication.processEvents()  # Update UI
            
            combined, contours = findcontours(
                self.injury_img, self.cropped_control, 
                self.threshvalue, self.linestep
            )
            
            self.result_img = combined
            self.contours = contours
            
            pixmap = cvimg_to_qpixmap(combined)
            self.result_label.setPixmap(pixmap)
            
            # Enable save buttons and update their styles
            self.save_btn.setEnabled(True)
            self.save_btn.setStyleSheet(self.get_button_style())
            self.save_contour_btn.setEnabled(True)
            self.save_contour_btn.setStyleSheet(self.get_button_style())
            
            # Show processing info
            contour_count = len(contours) if contours else 0
            QMessageBox.information(self, "Analysis Complete", 
                                  f"✅ Analysis completed successfully!\n\n"
                                  f"📊 Detected {contour_count} contour region(s)\n"
                                  f"🎯 Threshold: {self.threshvalue}\n"
                                  f"📏 Line density: {self.linestep}\n\n"
                                  f"Results are ready for export.")
            
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to process images: {str(e)}")
            print(f"Processing error: {e}")
            traceback.print_exc()
            self.result_label.setText("❌ Processing failed.\nCheck console for details.")

    def save_result(self):
        if self.result_img is None:
            return
            
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Analysis Results", "robodoc_injury_analysis.png", 
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            if path:
                success = cv2.imwrite(path, self.result_img)
                if success:
                    QMessageBox.information(self, "Export Successful", 
                                          f"✅ Analysis results saved successfully!\n\nLocation: {path}")
                else:
                    QMessageBox.warning(self, "Export Error", "❌ Failed to save the analysis results.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save results: {str(e)}")

    def save_selected_contours(self):
        if self.injury_img is None or self.contours is None:
            return
            
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Contours", "robodoc_injury_contours.png", 
                "PNG Files (*.png);;All Files (*)"
            )
            if path:
                save_selected_contours(self.injury_img, self.contours, path)
                QMessageBox.information(self, "Export Successful", 
                                      f"✅ Contours exported successfully!\n\nLocation: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export contours: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    
    # Set application properties for professional appearance
    app.setApplicationName("RoboDoc v1")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Medical Imaging Solutions")
    
    try:
        window = InjuryDetectionApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()