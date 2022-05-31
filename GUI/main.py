import sys
from PySide2.QtWidgets import QApplication
from ui import UI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = UI()
    mainWindow.show()
    sys.exit(app.exec_())
