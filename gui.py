import sys
import os
import subprocess
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QUrl
)
from PyQt5.QtGui import (
    QPixmap, QImage, QDesktopServices, QIcon, QFont
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QScrollArea, QGridLayout, QProgressDialog, QSpacerItem,
    QSizePolicy, QFrame
)
from PIL import Image

# ----------------------------------------------------
#   Import your indexing/search logic from main.py
# ----------------------------------------------------
from main import index_images, search_images, configure_logging


SIMILARITY_THRESHOLD = 0.22

# ----------------------------------------------------
#               Worker Threads
# ----------------------------------------------------
class IndexWorker(QThread):
    """Indexes images in a given folder in a background thread."""
    finished = pyqtSignal(object, str)  # (embeddings, folder_path)
    error = pyqtSignal(str)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
    
    def run(self):
        try:
            embeddings = index_images(self.folder_path)
            self.finished.emit(embeddings, self.folder_path)
        except Exception as e:
            self.error.emit(str(e))

class SearchWorker(QThread):
    """Performs search on the already indexed embeddings in a background thread."""
    finished = pyqtSignal(object, str)  # (results, query)
    error = pyqtSignal(str)
    
    def __init__(self, query, embeddings):
        super().__init__()
        self.query = query
        self.embeddings = embeddings
    
    def run(self):
        try:
            results = search_images(
                self.query,
                self.embeddings,
                top_k=len(self.embeddings),
                threshold=SIMILARITY_THRESHOLD,
            )
            self.finished.emit(results, self.query)
        except Exception as e:
            self.error.emit(str(e))

# ----------------------------------------------------
#         ClickableLabel for clickable thumbnails
# ----------------------------------------------------
class ClickableLabel(QLabel):
    """A QLabel that emits a signal when clicked."""
    clicked = pyqtSignal(str)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)

# ----------------------------------------------------
#          ImageCard Widget
# ----------------------------------------------------
class ImageCard(QWidget):
    """
    A widget that holds:
      - A clickable image preview.
      - A title label with an arrow icon button beside it.
      - An optional similarity label.
    Clicking the arrow icon (or the image) will trigger navigation to the image location.
    """
    clicked = pyqtSignal(str)  # Signal to open the image location.

    def __init__(self, image_path="", title="Image Name", similarity=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.title = title
        self.similarity = similarity
        
        # Main layout.
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Clickable image label.
        self.image_label = ClickableLabel(self.image_path)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("ImagePreview")
        self.image_label.clicked.connect(self.on_image_clicked)
        
        # Load thumbnail (or fallback).
        pixmap = self._load_thumbnail(self.image_path)
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label)
        
        # Title area with arrow icon.
        title_layout = QHBoxLayout()
        
        # Title label.
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("ImageTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_layout.addWidget(self.title_label)
        
        # Arrow icon button.
        self.arrow_button = QPushButton()
        self.arrow_button.setObjectName("ArrowButton")
        self.arrow_button.setIcon(QIcon("arrow.png"))
        self.arrow_button.setIconSize(QSize(16, 16))
        self.arrow_button.setFlat(True)  # Remove the button border.
        self.arrow_button.clicked.connect(self.on_arrow_clicked)
        title_layout.addWidget(self.arrow_button)
        
        layout.addLayout(title_layout)
        
        # Optional: similarity label.
        if self.similarity is not None:
            self.similarity_label = QLabel(f"Similarity: {self.similarity:.4f}")
            self.similarity_label.setAlignment(Qt.AlignCenter)
            self.similarity_label.setStyleSheet("color: #BBBBBB; font-size: 12px;")
            layout.addWidget(self.similarity_label)
        
        # Styling for the card.
        self.setObjectName("ImageCardFrame")
        self.setLayout(layout)
    
    def on_image_clicked(self, path):
        """Relay the click signal so the parent can handle navigation."""
        self.clicked.emit(path)
    
    def on_arrow_clicked(self):
        """Triggered when the arrow icon is clicked. Emits the same signal."""
        self.clicked.emit(self.image_path)
    
    def _load_thumbnail(self, path):
        """Load and scale the image for display."""
        if os.path.exists(path):
            try:
                pil_image = Image.open(path)
                # Increase the thumbnail size for a larger preview
                pil_image.thumbnail((600, 600))
                pil_image = pil_image.convert("RGBA")
                data = pil_image.tobytes("raw", "RGBA")
                qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
            except Exception:
                pixmap = QPixmap(400, 400)
                pixmap.fill(Qt.darkGray)
        else:
            pixmap = QPixmap(400, 400)
            pixmap.fill(Qt.darkGray)
        
        # Return a larger scaled pixmap for better visibility
        return pixmap.scaled(
            400, 400,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )



# ----------------------------------------------------
#                  Main Window
# ----------------------------------------------------
class SnapSeek(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SnapSeek")
        # Allow resizing/maximizing by not fixing the size
        self.resize(900, 500)
        self.setMinimumSize(900, 500)
        
        # Set the window icon to logo.png
        self.setWindowIcon(QIcon("logo.png"))
        
        # Will hold the image embeddings after indexing
        self.embeddings = None
        
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top section (logo, title, subtitle, select folder & search bar)
        self.init_top_section()
        
        # Scroll area for image grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # Set an object name so we can style it
        self.scroll_area.setObjectName("ResultsScrollArea")
        self.main_layout.addWidget(self.scroll_area)
        
        # Container for the grid
        self.grid_container = QWidget()
        self.grid_container.setObjectName("grid_container")
        self.image_grid_layout = QGridLayout(self.grid_container)
        self.image_grid_layout.setSpacing(20)
        self.scroll_area.setWidget(self.grid_container)
        
        # Keep references to cards to avoid garbage collection
        self.image_cards = []
        
        # Apply style sheet
        self.apply_styles()
    
    def init_top_section(self):
        """Create the top section with:
           - A center area for the logo, title, and subtitle (center aligned)
           - A row for the select folder button, search bar, and search button.
        """
        top_container = QVBoxLayout()
        
        # --- Title Area: Logo, App Title, and Subtitle (centered) ---
        title_container = QVBoxLayout()
        logo_title_row = QHBoxLayout()
        logo_title_row.addStretch(1)
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(logo_pixmap)
        logo_title_row.addWidget(self.logo_label)
        logo_title_row.addSpacing(5)  # reduced spacing between logo and title
        self.app_title = QLabel("SnapSeek")
        self.app_title.setObjectName("AppTitle")
        logo_title_row.addWidget(self.app_title)
        logo_title_row.addStretch(1)
        title_container.addLayout(logo_title_row)
        
        self.app_subtitle = QLabel("AI image search with cached indexing and production-style observability.")
        self.app_subtitle.setObjectName("AppSubtitle")
        self.app_subtitle.setAlignment(Qt.AlignCenter)
        title_container.addWidget(self.app_subtitle)
        top_container.addLayout(title_container)
        
        # --- Search Area Row: Select Folder button, Search bar, and Search button ---
        search_container = QHBoxLayout()
        
        # Select Folder button with icon (now on the left side)
        self.select_folder_btn = QPushButton(" Select Folder")
        self.select_folder_btn.setObjectName("SelectFolderButton")
        self.select_folder_btn.setIcon(QIcon("open-folder.png"))
        self.select_folder_btn.setIconSize(QSize(16, 16))
        self.select_folder_btn.clicked.connect(self.select_folder)
        search_container.addWidget(self.select_folder_btn)
        
        search_container.addSpacing(10)  # Optional spacing between the button and search bar
        
        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.setObjectName("SearchBar")
        self.search_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_container.addWidget(self.search_bar)
        
        # Search button
        self.search_btn = QPushButton()
        self.search_btn.setObjectName("SearchButton")
        self.search_btn.setToolTip("Search")
        self.search_btn.setIcon(QIcon("search_icon.png"))
        self.search_btn.setIconSize(QSize(20, 20))
        self.search_btn.clicked.connect(self.perform_search)
        search_container.addWidget(self.search_btn)
        
        top_container.addLayout(search_container)
        
        self.main_layout.addLayout(top_container)
    
    def apply_styles(self):
        """Apply a custom style sheet for a dark, modern UI that resembles your reference."""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0B0D1A, stop:1 #182135
                );
            }
            #AppTitle {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }
            #AppSubtitle {
                font-size: 14px;
                color: #cfcfcf;
            }
            #SelectFolderButton, #SearchButton {
                background-color: #2D3956;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            #SelectFolderButton:hover, #SearchButton:hover {
                background-color: #3B4A6A;
            }
            #SearchBar {
                background-color: #1D2535;
                color: #ffffff;
                border: 1px solid #3B4A6A;
                padding: 6px;
                font-size: 14px;
                border-radius: 4px;
            }
            /* Make the scroll area match the app color (transparent to let gradient show) */
            #ResultsScrollArea {
                background-color: #1D2535;
                border: none;
            }
            /* The container inside the scroll area also transparent, so it doesn't appear white */
            #grid_container {
                background-color: #1D2535;
            }
            #ImageCardFrame {
                background-color: #1E2638;
                border-radius: 8px;
                padding: 10px;
            }
            #ImageTitle {
                color: #ffffff;
                font-size: 13px;
            }
            #ImagePreview {
                border: 1px solid #3B4A6A;
                border-radius: 4px;
            }
        """)
    
    # ------------------------------------------------
    #         Folder Selection / Indexing
    # ------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.progress = QProgressDialog("Indexing images...", None, 0, 0, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.show()
            
            self.select_folder_btn.setEnabled(False)
            self.search_btn.setEnabled(False)
            self.search_bar.setEnabled(False)
            
            self.index_thread = IndexWorker(folder)
            self.index_thread.finished.connect(self.index_finished)
            self.index_thread.error.connect(self.worker_error)
            self.index_thread.start()
    
    def index_finished(self, embeddings, folder):
        self.embeddings = embeddings
        self.progress.cancel()
        self.select_folder_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.search_bar.setEnabled(True)
        
        num_images = len(embeddings) if embeddings else 0
        QMessageBox.information(self, "Folder Indexed", f"Indexed {num_images} image(s) from:\n{folder}")
    
    # ------------------------------------------------
    #               Perform Search
    # ------------------------------------------------
    def perform_search(self):
        query = self.search_bar.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a search query.")
            return
        if not self.embeddings:
            QMessageBox.warning(self, "No Images Indexed", "Please select an image folder first.")
            return
        
        self.progress = QProgressDialog("Searching...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        
        self.select_folder_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.search_bar.setEnabled(False)
        
        self.search_thread = SearchWorker(query, self.embeddings)
        self.search_thread.finished.connect(self.search_finished)
        self.search_thread.error.connect(self.worker_error)
        self.search_thread.start()
    
    def search_finished(self, results, query):
        self.progress.cancel()
        self.select_folder_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.search_bar.setEnabled(True)
        
        if not results:
            QMessageBox.information(
                self,
                "No Results",
                f"No images found with similarity greater than {SIMILARITY_THRESHOLD:.2f}.",
            )
        
        self.display_results(results, query)
    
    # ------------------------------------------------
    #            Error Handling
    # ------------------------------------------------
    def worker_error(self, error_msg):
        self.progress.cancel()
        QMessageBox.critical(self, "Error", error_msg)
        self.select_folder_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.search_bar.setEnabled(True)
    
    # ------------------------------------------------
    #           Display Search Results
    # ------------------------------------------------
    def display_results(self, results, query):
        # Clear existing items in the grid
        for i in reversed(range(self.image_grid_layout.count())):
            widget = self.image_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.image_cards.clear()
        
        # Optionally add a row at the top with a header label
        header = QLabel(f'Search Results for: "{query}"')
        header.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        self.image_grid_layout.addWidget(header, 0, 0, 1, 3)  # spanning 3 columns
        
        # Populate with ImageCards
        row = 1
        col = 0
        columns_per_row = 3
        
        for (img_path, score) in results:
            # Title: the file name
            title = os.path.basename(img_path)
            
            card = ImageCard(
                image_path=img_path,
                title=title,
                similarity=score
            )
            card.clicked.connect(self.open_image_location)
            
            self.image_grid_layout.addWidget(card, row, col)
            self.image_cards.append(card)
            
            col += 1
            if col >= columns_per_row:
                col = 0
                row += 1
    
    # ------------------------------------------------
    #     Open Image in File Explorer / Finder
    # ------------------------------------------------
    def open_image_location(self, image_path):
        if os.path.exists(image_path):
            if sys.platform == "win32":
                subprocess.run(["explorer", f"/select,{os.path.normpath(image_path)}"], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", image_path], check=False)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(image_path)))
        else:
            QMessageBox.warning(self, "File Not Found", f"Could not locate: {image_path}")

# ----------------------------------------------------
#                  Main Entry
# ----------------------------------------------------
def main():
    configure_logging()
    app = QApplication(sys.argv)
    window = SnapSeek()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
