# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ferSPIGMd.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1017, 659)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setLayoutDirection(Qt.LeftToRight)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.verticalLayout_2 = QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox = QGroupBox(self.widget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setLayoutDirection(Qt.LeftToRight)
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.label_3)

        self.model_cbox = QComboBox(self.groupBox)
        self.model_cbox.addItem("")
        self.model_cbox.setObjectName(u"model_cbox")
        self.model_cbox.setContextMenuPolicy(Qt.DefaultContextMenu)

        self.horizontalLayout_4.addWidget(self.model_cbox)


        self.verticalLayout_2.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.widget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.input_btn = QPushButton(self.groupBox_2)
        self.input_btn.setObjectName(u"input_btn")
        self.input_btn.setLayoutDirection(Qt.LeftToRight)

        self.verticalLayout_4.addWidget(self.input_btn)

        self.input_edit = QLineEdit(self.groupBox_2)
        self.input_edit.setObjectName(u"input_edit")
        self.input_edit.setAlignment(Qt.AlignCenter)
        self.input_edit.setReadOnly(True)

        self.verticalLayout_4.addWidget(self.input_edit)


        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.widget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_2)

        self.time_edit = QLineEdit(self.groupBox_3)
        self.time_edit.setObjectName(u"time_edit")
        self.time_edit.setAlignment(Qt.AlignCenter)
        self.time_edit.setReadOnly(True)

        self.horizontalLayout_3.addWidget(self.time_edit)

        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.label)

        self.expression_edit = QLineEdit(self.groupBox_3)
        self.expression_edit.setObjectName(u"expression_edit")
        self.expression_edit.setEnabled(True)
        self.expression_edit.setAlignment(Qt.AlignCenter)
        self.expression_edit.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.expression_edit)

        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_2)

        self.emoil_show = QLabel(self.groupBox_3)
        self.emoil_show.setObjectName(u"emoil_show")
        self.emoil_show.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.emoil_show)

        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 1)
        self.verticalLayout_5.setStretch(2, 3)

        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 2)

        self.horizontalLayout.addWidget(self.widget)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.img_show = QLabel(self.centralwidget)
        self.img_show.setObjectName(u"img_show")
        self.img_show.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.img_show)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")

        self.verticalLayout.addWidget(self.graphicsView)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(2, 5)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Model", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u" select model ", None))
        self.model_cbox.setItemText(0, QCoreApplication.translate("MainWindow", u"VggNet", None))

        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Input", None))
        self.input_btn.setText(QCoreApplication.translate("MainWindow", u"input image", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Result", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"    time    ", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u" expression ", None))
        self.emoil_show.setText("")
        self.img_show.setText("")
    # retranslateUi
