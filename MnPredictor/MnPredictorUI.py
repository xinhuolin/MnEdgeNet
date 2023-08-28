#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from UI_files.Atom_Seg_Ui import Ui_MainWindow
import pandas
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import torch
import EELSModel2 as m
import EELSModel3 as m2
import EELSModel4 as m3
from matplotlib.backends.backend_qt5agg import FigureCanvas


def truncate(num):
    return math.floor(num * 10.0) / 10.0

def transform(x, f, start, end):
    if x < start:
        return 0
    elif x > end:
        return f(truncate(end)) - f(truncate(start)+0.1)
    else:
        return f(x) - f(truncate(start)+0.1)

class Code_MainWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        super(Code_MainWindow, self).__init__()

        self.setupUi(self)
        self.open.clicked.connect(self.BrowseFolder)
        self.load.clicked.connect(self.LoadModel)
        # self.se_num.valueChanged.connect(self.Denoise)

        # self.circle_detect.clicked.connect(self.CircleDetect)

        # self.revert.clicked.connect(self.RevertAll)

        self.clear.clicked.connect(self.Clear)

        self.__curdir = os.getcwd()  # current directory

        self.ori_image = None
        self.ori_content = None  # original image, PIL format
        self.output_image = None  # output image of model, PIL format
        self.ori_markers = None  # for saving usage, it's a rgb image of original, and with detection result on it
        self.out_markers = None  # for saving usage, it's a rgb image of result after denoising, and with detection result on it
        self.model_output_content = None  # 2d array of model output

        self.result = None
        self.denoised_image = None
        self.props = None

        self.imarray_original = None

        self.files = {}
        self.lastDirectory = None

        self.names = ['model1 (uniform scaling)',
                      'model2 (trained without noise)',
                      'model3 (correct scaling)',
                      'model4 (fixed mean and std, correct scaling)',
                      'model5 (Used mean and std of each spectrum)',
                      'model6 (Increased Mn metal appearance)',
                      'model7_1 (no metal)',
                      'model8_1 (with metal)']

        self.__model_dir = "MnPredictor\model_weights"
        self.__models = {
            self.names[0]: os.path.join(self.__model_dir, 'EELS4.pt'),
            self.names[1]: os.path.join(self.__model_dir, 'EELS2.pt'),
            self.names[2]: os.path.join(self.__model_dir, 'EELS1.pt'),
            self.names[3]: os.path.join(self.__model_dir, 'EELS.pt'),
            self.names[4]: os.path.join(self.__model_dir, 'EELS6.pt'),
            self.names[5]: os.path.join(self.__model_dir, 'EELSModel6.pt'),
            self.names[6]: os.path.join(self.__model_dir, 'EELS7.pt'),
            self.names[7]: os.path.join(self.__model_dir, 'EELS8.pt'),
        }
        self.mean = {self.names[0]: os.path.join(self.__model_dir, 'mean3.csv'),
                      self.names[1]: os.path.join(self.__model_dir, 'mean2.csv'),
                     self.names[2]: os.path.join(self.__model_dir, 'mean4.csv'),
                     self.names[3]: os.path.join(self.__model_dir, 'mean.csv')}
        self.std = {self.names[0]: os.path.join(self.__model_dir, 'std3.csv'),
                      self.names[1]: os.path.join(self.__model_dir, 'std2.csv'),
                    self.names[2]: os.path.join(self.__model_dir, 'std4.csv'),
                    self.names[3]: os.path.join(self.__model_dir, 'std.csv')}
        # from torch.cuda import is_available
        # self.use_cuda.setChecked(is_available())
        # self.use_cuda.setDisabled(not is_available())

        self.imagePath_content = None
        self.ori.setText('Files to be Analyzed: \n')
        self.model_output.setText('Prediction Output: \n')
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Energy Loss (eV)')
        self.plotWidget = FigureCanvas(self.fig)
        self.plotWidget.setFixedSize(850, 400)
        # self.plotWidget.move(20, 580)
        self.preprocess.addWidget(self.plotWidget)
        self.plotWidget.draw()

    def BrowseFolder(self):
        if self.lastDirectory is None:
            path, _ = QFileDialog.getOpenFileName(self,
                                                  "Open File",
                                                  os.getcwd(),
                                                  "Text Files (*.csv)")
        else:
            path, _ = QFileDialog.getOpenFileName(self,
                                                  "Open File",
                                                  self.lastDirectory,
                                                  "Text Files (*.csv)")
        self.imagePath_content = self.imagePath_content if not path else path
        if self.imagePath_content:
            args = path.split('\\')
            self.lastDirectory = '\\'.join(args[:-1])
            self.imagePath.setText(self.imagePath_content)
            file_name = os.path.basename(self.imagePath_content)
            _, suffix = os.path.splitext(file_name)
            # # file_name = self.imagePath_content.split('/')[-1]
            # # suffix = '.' + file_name.split('.')[-1]
            # if suffix == '.ser':
            #     from file_readers.ser_lib.serReader import serReader
            #     ser_data = serReader(self.imagePath_content)
            #     ser_array = np.array(ser_data['imageData'], dtype='float64')
            #     self.imarray_original = ser_array
            #     ser_array = (map01(ser_array) * 255).astype('uint8')
            #     self.ori_image = Image.fromarray(ser_array, 'L')
            # elif suffix == '.dm3':
            #     from file_readers import dm3_lib as dm3
            #     data = dm3.DM3(self.imagePath_content).imagedata
            #     self.imarray_original = np.array(data)
            #     data = np.array(data, dtype='float64')
            #     data = (map01(data) * 255).astype('uint8')
            #     self.ori_image = Image.fromarray(data, mode='L')
            # elif suffix == '.tif':
            #     im = Image.open(self.imagePath_content).convert('L')
            #     self.imarray_original = np.array(im, dtype='float64')
            #     self.ori_image = Image.fromarray((map01(self.imarray_original) * 255).astype('uint8'), mode='L')
            # else:
            #     self.ori_image = Image.open(self.imagePath_content).convert('L')
            #     self.imarray_original = np.array(self.ori_image)
            #
            # self.width, self.height = self.ori_image.size
            # pix_image = PIL2Pixmap(self.ori_image)
            # pix_image.scaled(self.ori.size(), QtCore.Qt.KeepAspectRatio)
            # self.ori.setPixmap(pix_image)
            # self.ori.show()
            # self.ori_content = self.ori_image
            try:
                if self.header.isChecked():
                    df = pandas.read_csv(self.imagePath_content)
                else:
                    df = pandas.read_csv(self.imagePath_content, header=None)
                matrix = df.to_numpy()
                matrix = np.transpose(matrix)
                func = interpolate.interp1d(matrix[0], matrix[1])
                xval = np.arange(630.6, 669.4, 0.1)
                data = [transform(x, func, matrix[0][0], matrix[0][-1]) for x in xval]
                data = np.array(data)
                data = data / max(data)
                self.files[file_name] = data
                self.ori.setText(self.ori.text() + file_name + '\n')
                self.ax.plot(xval, data, label=file_name)
                self.ax.legend()
                self.plotWidget.draw()
            except:
                QMessageBox.warning(self, "Error: Could not read file", self.tr("Could not read file. Please double check header and/or column name"))
        self.imagePath_content = None

    def __load_model(self):
        if not bool(self.files):
            raise Exception("No file is selected")
        # self.cuda = self.use_cuda.isChecked()
        model_path = os.path.join(self.__curdir, self.__models[self.model_name])

        if self.model_name == self.names[3] or self.model_name == self.names[4]\
                or self.model_name == self.names[6]:
            model = m2.Model(0.0001, torch.nn.MSELoss(), 388, 3)
        elif self.model_name == self.names[5] or self.model_name == self.names[7]:
            model = m3.Model(0.0001, torch.nn.MSELoss(), 388, 4)
        else:
            model = m.Model(0.0001, torch.nn.MSELoss(), 388, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        for key, value in self.files.items():
            validation_set = torch.tensor(value, dtype=torch.float32)
            if self.model_name == self.names[4] or self.model_name == self.names[5]\
                    or self.model_name == self.names[6]\
                    or self.model_name == self.names[7]:
                mean = validation_set.mean(dim=0, keepdim=True)
                std = validation_set.std(dim=0, keepdim=True)
            else:
                df = pandas.read_csv(os.path.join(self.__curdir, self.mean[self.model_name]), header=None)
                data = df.to_numpy()
                mean = torch.tensor(data, dtype=torch.float32)

                df = pandas.read_csv(os.path.join(self.__curdir, self.std[self.model_name]), header=None)
                data = df.to_numpy()
                std = torch.tensor(data, dtype=torch.float32)
            standardize = (validation_set - mean) / std
            result = model.forward(standardize)
            result = result * 100
            for item in result:
                temp = item.data.tolist()
                for i in range(len(temp)):
                    temp[i] = round(temp[i], 2)
                # print(key + ': ' + str(temp))
                state = round((temp[0] * 2 + temp[1] * 3 + temp[2] * 4) / 100.0, 2)
                self.model_output.setText(
                    self.model_output.text() + key + ': ' + str(temp) + '     Oxidation: ' + str(state) + '\n')
                # print(round((temp[0] * 2 + temp[1] * 3 + temp[2] * 4) / 100.0, 2))


        # if self.change_size.currentText() == 'Down sample by 2':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width // 2, self.height // 2), Image.BILINEAR)
        # elif self.change_size.currentText() == 'Up sample by 2':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width * 2, self.height * 2), Image.BICUBIC)
        # elif self.change_size.currentText() == 'Down sample by 3':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width // 3, self.height // 3), Image.BILINEAR)
        # elif self.change_size.currentText() == 'Up sample by 3':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width * 3, self.height * 3), Image.BICUBIC)
        # elif self.change_size.currentText() == 'Down sample by 4':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width // 4, self.height // 4),
        #                                              Image.BILINEAR)
        # elif self.change_size.currentText() == 'Up sample by 4':
        #     self.width, self.height = self.ori_image.size
        #     self.ori_content = self.ori_image.resize((self.width * 4, self.height * 4),
        #                                              Image.BICUBIC)
        # else:
        #     self.ori_content = self.ori_image
        #
        # pix_image = PIL2Pixmap(self.ori_content)
        # pix_image.scaled(self.ori.size(), QtCore.Qt.KeepAspectRatio)
        # self.ori.setPixmap(pix_image)
        # self.ori.show()
        #
        # self.width, self.height = self.ori_content.size

        # if self.header.isChecked():
        #
        #     if self.height > 512 and self.height <= 1024:
        #         blk_row = 2
        #     else:
        #         if self.height > 1024:
        #             blk_row = 4
        #         else:
        #             blk_row = 1
        #
        #     if self.width > 512 and self.width <= 1024:
        #         blk_col = 2
        #     else:
        #         if self.width > 1024:
        #             blk_col = 4
        #         else:
        #             blk_col = 1
        # else:
        #     blk_col = 1
        #     blk_row = 1
        #
        # self.result = np.zeros((self.height, self.width)) - 100
        #
        # for r in range(0, blk_row):
        #     for c in range(0, blk_col):
        #         inner_blk, outer_blk = GetIndexRangeOfBlk(self.height, self.width, blk_row, blk_col, r, c,
        #                                                   over_lap=int(self.width * 0.01))
        #         temp_image = self.ori_content.crop((outer_blk[0], outer_blk[1], outer_blk[2], outer_blk[3]))
        #         temp_result = load_model(model_path, temp_image, self.cuda, self.set_iter.value())
        #         #                temp_result = map01(temp_result)
        #         self.result[outer_blk[1]: outer_blk[3], outer_blk[0]: outer_blk[2]] = np.maximum(temp_result,
        #                                                                                          self.result[
        #                                                                                          outer_blk[1]:outer_blk[
        #                                                                                              3], outer_blk[0]:
        #                                                                                                  outer_blk[2]])
        #
        # self.result[self.result < 0] = 0
        # self.model_output_content = map01(self.result)
        # self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype(
        #     'uint8')
        # self.output_image = Image.fromarray((self.model_output_content), mode='L')
        # pix_image = PIL2Pixmap(self.output_image)
        # pix_image.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
        # self.model_output.setPixmap(pix_image)
        # self.model_output.show()
        # del temp_image
        # del temp_result

    def LoadModel(self):
        self.model_name = self.modelPath.currentText()
        if not bool(self.files):
            QMessageBox.warning(self, "Please select a csv file", self.tr("Please select a csv file"))
            return
        self.__load_model()
        # self.Denoise()

    # def Denoise(self):
    #     radius = self.se_num.value()
    #     """changes should be done on the kernel generation"""
    #     kernel = disk(radius)
    #
    #     if self.denoise_method.currentText == 'Opening':
    #         self.denoised_image = opening(self.model_output_content, kernel)
    #     else:
    #         self.denoised_image = erosion(self.model_output_content, kernel)
    #
    #     temp_image = Image.fromarray(self.denoised_image, mode='L')
    #
    #     pix_image = PIL2Pixmap(temp_image)
    #     self.preprocess.setPixmap(pix_image)
    #     self.preprocess.show()
    #     del temp_image

    # def CircleDetect(self):
    #     if not self.imagePath_content:
    #         QMessageBox.warning(self, "Please select a file", self.tr("Please select a file"))
    #         return
    #     elevation_map = sobel(self.denoised_image)
    #
    #     from scipy import ndimage as ndi
    #     markers = np.zeros_like(self.denoised_image)
    #     if self.set_thre.isChecked() and self.thre.text():
    #         max_thre = int(self.thre.text()) * 2.55
    #     else:
    #         max_thre = 100
    #
    #     min_thre = 30
    #     markers[self.denoised_image < min_thre] = 1
    #     markers[self.denoised_image > max_thre] = 2
    #
    #     seg_1 = watershed(elevation_map, markers)
    #
    #     filled_regions = ndi.binary_fill_holes(seg_1 - 1)
    #
    #     label_objects, nb_labels = ndi.label(filled_regions)
    #
    #     self.props = regionprops(label_objects)
    #
    #     self.out_markers = Image.fromarray(np.dstack((self.denoised_image, self.denoised_image, self.denoised_image)),
    #                                        mode='RGB')
    #
    #     ori_array = np.array(self.ori_content)
    #     self.ori_markers = Image.fromarray(np.dstack((ori_array, ori_array, ori_array)), mode='RGB')
    #
    #     del elevation_map
    #     del markers, seg_1, filled_regions, label_objects, nb_labels
    #
    #     draw_out = ImageDraw.Draw(self.out_markers)
    #     draw_ori = ImageDraw.Draw(self.ori_markers)
    #
    #     for p in self.props:
    #         c_y, c_x = p.centroid
    #         draw_out.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height]),
    #                           min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])],
    #                          fill='red', outline='red')
    #         draw_ori.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height]),
    #                           min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])],
    #                          fill='red', outline='red')
    #
    #     pix_image = PIL2Pixmap(self.out_markers)
    #     self.preprocess.setPixmap(pix_image)
    #     self.preprocess.show()
    #     pix_image = PIL2Pixmap(self.ori_markers)
    #     self.detect_result.setPixmap(pix_image)
    #     self.detect_result.show()

    #        del props

    # def RevertAll(self):
    #     self.model_output.clear()
    #     self.se_num.setValue(0)
    #     self.preprocess.clear()
    #     # self.detect_result.clear()
    #     del self.result
    #
    #     self.result = None

    # def GetSavePath(self):
    #     file_name = os.path.basename(self.imagePath_content)
    #     _, suffix = os.path.splitext(file_name)
    #     if suffix in ['.ser', '.dm3', '.tif']:
    #         name_no_suffix = file_name.replace(suffix, '')
    #         suffix = '.png'
    #     else:
    #         name_no_suffix = file_name.replace(suffix, '')
    #
    #     if not self.change_size.currentText() == 'Do Nothing':
    #         name_no_suffix = name_no_suffix + '_' + self.change_size.currentText()
    #     has_content = True
    #
    #     if self.auto_save.isChecked():
    #         save_path = os.path.join(self.__curdir, name_no_suffix)
    #     else:
    #         path = QFileDialog.getExistingDirectory(self, "save", self.__curdir,
    #                                                 QFileDialog.ShowDirsOnly
    #                                                 | QFileDialog.DontResolveSymlinks)
    #         if not path:
    #             has_content = False
    #         save_path = os.path.join(path, name_no_suffix)
    #
    #     if has_content:
    #         if not exists(save_path):
    #             os.mkdir(save_path)
    #         temp_path = os.path.join(save_path, name_no_suffix)
    #     else:
    #         temp_path = None
    #
    #     return temp_path, suffix

    # def Save(self):
    #     if not self.imagePath_content:
    #         QMessageBox.warning(self, "Please select a file", self.tr("Please select a file"))
    #         return
    #     opt = self.save_option.currentText()
    #     _path, suffix = self.GetSavePath()
    #     if _path is None:
    #         return
    #     new_save_name = _path + '_output_' + self.model_name + '.mat'
    #     scio.savemat(new_save_name, {'result': self.result})
    #     new_save_name = _path + '_ori_' + self.model_name + '.mat'
    #     scio.savemat(new_save_name, {'origin': self.imarray_original})
    #
    #     if not _path:
    #         return
    #
    #     if opt == 'Model output':
    #         new_save_name = _path + '_output_' + self.model_name + suffix
    #         self.output_image.save(new_save_name)
    #
    #     if opt == 'Original image with markers':
    #         new_save_name = _path + '_origin_' + self.model_name + suffix
    #         self.ori_markers.save(new_save_name)
    #
    #     if opt == 'Four-panel image':
    #         new_save_name = _path + '_four_panel_' + self.model_name + suffix
    #         im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
    #         im_save.paste(self.ori_content, (0, 0))
    #         im_save.paste(self.output_image, (self.width + 2, 0))
    #         im_save.paste(self.ori_markers, (0, self.height + 2))
    #         im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
    #         im_save.save(new_save_name)
    #         del im_save
    #
    #     if opt == 'Atom positions':
    #         new_save_name = _path + '_pos_' + self.model_name + '.txt'
    #         file = open(new_save_name, 'w')
    #         for p in self.props:
    #             c_y, c_x = p.centroid
    #             min_row, min_col, max_row, max_col = p.bbox
    #             c_y_int = int(min(max(round(c_y), 0), self.height))
    #             c_x_int = int(min(max(round(c_x), 0), self.width))
    #             locations = [str(i) for i in
    #                          (c_y, c_x, min_row, min_col, max_row, max_col, self.result[c_y_int, c_x_int])]
    #             file.write(",".join(locations))
    #             file.write("\n")
    #         file.close()
    #
    #     if opt == 'Save ALL':
    #         new_save_name = _path + suffix
    #         self.ori_content.save(new_save_name)
    #         new_save_name = _path + '_output_' + self.model_name + suffix
    #         self.output_image.save(new_save_name)
    #         new_save_name = _path + '_origin_' + self.model_name + suffix
    #         self.ori_markers.save(new_save_name)
    #         new_save_name = _path + '_four_panel_' + self.model_name + suffix
    #         im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
    #         im_save.paste(self.ori_content, (0, 0))
    #         im_save.paste(self.output_image, (self.width + 2, 0))
    #         im_save.paste(self.ori_markers, (0, self.height + 2))
    #         im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
    #         im_save.save(new_save_name)
    #         del im_save
    #         new_save_name = _path + '_pos_' + self.model_name + '.txt'
    #         file = open(new_save_name, 'w')
    #         for p in self.props:
    #             c_y, c_x = p.centroid
    #             min_row, min_col, max_row, max_col = p.bbox
    #             c_y_int = int(min(max(round(c_y), 0), self.height))
    #             c_x_int = int(min(max(round(c_x), 0), self.width))
    #             locations = [str(i) for i in
    #                          (c_y, c_x, min_row, min_col, max_row, max_col, self.result[c_y_int, c_x_int])]
    #             file.write(",".join(locations))
    #             file.write("\n")
    #         file.close()

    # def drawPoint(self, event):
    #     self.pos = event.pos()
    #     self.update()

    def release(self):
        self.model_output.clear()
        # self.se_num.setValue(0)
        # self.preprocess.clear()
        # self.detect_result.clear()
        self.ori.clear()
        # del self.props
        # del self.output_image
        # del self.ori_markers
        # del self.out_markers
        return

    def Clear(self):
        self.ax.cla()
        self.plotWidget.draw()
        self.ori.setText('Files to be Analyzed: \n')
        self.model_output.setText('Prediction Output: \n')
        self.files.clear()
        return

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            self.release()
            event.accept()


qtCreatorFile = os.path.join("MnPredictor", "UI_files", "AtomSeg_V1.ui")

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Code_MainWindow()
    window.show()
    sys.exit(app.exec_())
