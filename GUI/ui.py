import os
import cv2
import torch
import time
import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import QMainWindow, QFileDialog
from PySide2.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Ui_window import Ui_MainWindow
from model.VggNet import Vgg
from recognition import predict_expression


class UI(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(UI, self).__init__(parent)
        self.setupUi(self)  # import ui interface
        self.cwd = os.getcwd()  # get current path
        self.models = {'VggNet': Vgg}
        self.checkpoints = {'VggNet': os.path.join(self.cwd, 'params', 'VGGNet')}
        self.emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.init_ui()

    def init_ui(self):
        self.input_btn.clicked.connect(self.select_input)

    def select_input(self):
        self.img_path, self.img_type = QFileDialog.getOpenFileNames(self, "Please select an image",
                                                                    self.cwd, "image Files (*.jpg *.png *.jpeg)")
        if self.img_path is not None and self.img_path != []:
            start = time.time()
            self.input_edit.setText(self.img_path[0])

            model = self.models[self.model_cbox.currentText()]()
            checkpoint = torch.load(self.checkpoints[self.model_cbox.currentText()])

            if self.model_cbox.currentText() == 'VggNet':
                model.load_state_dict(checkpoint['params'], strict=False)
            model = model.eval()

            img, faces, face_list = predict_expression(self.img_path[0], self.model_cbox.currentText())
            if len(faces) != 0:
                i = 0
                emoil_list = list()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                    outputs = model(face_list[i])
                    i += 1
                    outputs = torch.sum(outputs, dim=0) / 10
                    label_index = np.argmax(outputs.detach().numpy(), axis=0)
                    probability_list = torch.nn.functional.softmax(outputs, dim=0)
                    # print(probability_list)
                    emoil = self.emotion_mapping[label_index]
                    img = self.cv2_img_add_text(img, emoil, x + 10, y + 10, (255, 0, 0), 20)
                    emoil_list.append(emoil)

                result = ''
                for string in emoil_list:
                    result = result + string + ' '
                self.expression_edit.setText(result)
                self.show_img(img)
                self.show_emoil(emoil)
                self.show_bars(probability_list.detach().numpy())
                output_path = os.path.join(os.getcwd(), 'output', 'result.png')
                cv2.imwrite(output_path, img)

            else:
                self.show_img(img)
                emoil = 'None'
                self.show_emoil(emoil)
                self.expression_edit.setText(QtCore.QCoreApplication.translate("Form", "no result"))
                self.graphicsView.close()

            end = time.time()
            time_spend = end - start
            self.time_edit.setText(str(round(time_spend, 2)))

    def show_img(self, img):
        show = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (400, 400))
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.img_show.setPixmap(QPixmap.fromImage(showImage))
        self.img_show.show()
        # self.img_show.setScaledContents(True)  # Make the image adapt to the label size

    def cv2_img_add_text(self, img, text, left, top, text_color=(255, 255, 255), text_size=20):
        """
        :param img:
        :param text:
        :param left:
        :param top:
        :param text_color:
        :param text_size
        :return:
        """
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype(
            "./assets/simsun.ttc", text_size, encoding="utf-8")  # Use Arial
        draw.text((left, top), text, text_color, font=font_text)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def show_emoil(self, emotion):
        # 显示emoji
        if emotion != 'None':
            img = cv2.imread(os.path.join(self.cwd, 'icons', emotion+'.png'))
            frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (50, 50))
            self.emoil_show.setPixmap(QtGui.QPixmap.fromImage(
                QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                             QtGui.QImage.Format_RGB888)))
        else:
            self.emoil_show.setText(QtCore.QCoreApplication.translate("Form", "no result"))

    def show_bars(self, possbility):
        dr = MyFigureCanvas()
        dr.draw_(possbility)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        self.axes.bar(x, possibility, align='center')