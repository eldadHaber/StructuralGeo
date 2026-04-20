import os
import sys

os.environ["QT_API"] = "pyqt5"

from file_manager_gui import FileManagerGUI  # Import the file manager GUI class
from plotter import ModelPlotter  # Ensure this import matches the file location
from qtpy import QtCore, QtGui, QtWidgets
from toolbar import ToolBarWidget  # Ensure this import matches the file location

from geogen.filemanagement import FileManager


class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, show=True, base_dir=None):
        super(MyMainWindow, self).__init__(parent)

        # Set window title and icon
        self.setWindowTitle("GeoModel Viewer")
        icon_path = os.path.join(os.path.dirname(__file__), "large_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        else:
            print("Icon file not found")

        # Create a plotter, toolbar
        self.plotter = ModelPlotter(self)
        self.filemanager = FileManager(base_dir=base_dir)
        self.file_manager_gui = FileManagerGUI(self, self.filemanager)  # Pass the file manager to the GUI
        self.toolbar = ToolBarWidget(self, plotter=self.plotter, file_manager=self.filemanager)

        self._init_layout()
        self._init_filemenu()
        #      self.toolbar.add_shortcuts(self)

        if show:
            self.show()

        # Populate the file tree
        self.file_manager_gui.populate_file_tree()

    def _init_layout(self):
        # Set up the main layout
        self.central_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Create a splitter to hold the file tree and plotter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Add the file tree from file_manager_gui
        self.splitter.addWidget(self.file_manager_gui.file_tree)
        self.splitter.addWidget(self.plotter.frame)

        # Set the splitter to have an adjustable handle
        self.splitter.setSizes([250, 750])  # Initial sizes for the file tree and plotter
        self.splitter.setStretchFactor(1, 1)  # Allow the plotter to expand

        # Add the splitter to the main layout
        self.main_layout.addWidget(self.splitter)

        # Create a horizontal layout for the toolbar
        self.toolbar_layout = QtWidgets.QHBoxLayout()

        # Add a spacer to push the toolbar to the right
        self.toolbar_spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.toolbar_layout.addItem(self.toolbar_spacer)
        self.toolbar_layout.addWidget(self.toolbar)

        # Set toolbar size policy to fixed height
        self.toolbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Add the toolbar layout to the main layout
        self.main_layout.addLayout(self.toolbar_layout)

        # Set the central widget
        self.setCentralWidget(self.central_widget)

    def _init_filemenu(self):
        # Simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")

        # Add 'Select Folder' option in the menu
        selectFolderAction = QtWidgets.QAction("Select Models Folder", self)
        selectFolderAction.triggered.connect(self.file_manager_gui.select_folder)
        fileMenu.addAction(selectFolderAction)

        # Add 'Set Resolution' option in the menu
        setResolutionAction = QtWidgets.QAction("Set Resolution", self)
        setResolutionAction.triggered.connect(self.set_resolution)
        fileMenu.addAction(setResolutionAction)

        # Final option to exit the application
        exitButton = QtWidgets.QAction("Exit", self)
        exitButton.setShortcut("Ctrl+Q")
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

    def set_resolution(self):
        resolution, ok = QtWidgets.QInputDialog.getInt(
            self, "Set Resolution", "Enter resolution (2-256):", min=2, max=256
        )
        if ok:
            self.plotter.resolution = (resolution, resolution, resolution)
            print(f"Resolution set to: {self.plotter.resolution}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MyMainWindow(base_dir=os.getcwd())  # Pass the base directory to the main window')
    sys.exit(app.exec_())
