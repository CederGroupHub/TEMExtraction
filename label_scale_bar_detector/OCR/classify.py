from PIL import Image, ImageFilter, ImageEnhance
import PIL.ImageOps
import pytesseract
import time
import re
import os
import subprocess

class OCRBase:
    unsuccessful_return = None # What to return when unsuccessful
    tp = None # "scale" or "label"
    srcnn_path = None # path to srcnn output images

    def image2string(self, img):
        # Define in child class
        raise NotImplementedError

    def success(self, st):
        raise NotImplementedError

    def postprocess(self, st):
        return st

    def remove_spaces(self, st):
        return st.strip()

    def recognize_text(self, path: str):
        filename = path.split('/')[2]

        img = Image.open(path).convert('L')
        #print("image opened")
        img_inv = PIL.ImageOps.invert(img)

        normal = self.remove_spaces(self.image2string(img))
        inv = self.remove_spaces(self.image2string(img_inv))
        if self.success(normal):
            return self.postprocess(normal)
        elif self.success(inv):
            return self.postprocess(inv)
        else:
            try:
                subprocess.check_call('python label_scale_bar_detector/OCR/SRCNN-pytorch/test.py'
                                ' --weights-file "label_scale_bar_detector/OCR/SRCNN-pytorch/weights/srcnn_x4.pth"'
                                ' --image-file "{}"'
                                ' --type {}'
                                ' --scale {}'.format(path, self.tp, 4), shell=True)
            except:
                print("subprocess check_call failed!")
                return self.unsuccessful_return
            img_SRCNN = Image.open(os.path.join(self.srcnn_path, filename))
            img_SRCNN_inv = PIL.ImageOps.invert(img_SRCNN)

            SRCNN = self.remove_spaces(self.image2string(img_SRCNN))
            SRCNN_inv = self.remove_spaces(self.image2string(img_SRCNN_inv))

            if self.success(SRCNN):
                return self.postprocess(SRCNN)
            elif self.success(SRCNN_inv):
                return self.postprocess(SRCNN_inv)
            else:
                return self.unsuccessful_return

class OCRScale(OCRBase):
    unsuccessful_return = (None, None)
    tp = 'scale'
    srcnn_path = "extracted_data_single/Scales_SRCNN/"

    def image2string(self, img):
        return pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=uUnmM1234567890. --psm 7')

    def postprocess(self, st):
        if "nm" in st.lower():
            unit = "nm"
        elif "um" in st.lower():
            unit = "um"

        number = re.findall(r'\+*-*\d+', st)[0]
        return number, unit

    def success(self, st):
        if "nm" in st.lower():
            unit = "nm"
        elif "um" in st.lower():
            unit = "um"
        else:
            return False

        number = re.findall(r'\+*-*\d+', st)
        if len(number) == 0:
            return False
        else:
            if (float(number[0]) != 0 and float(number[0]) <= 9) or (float(number[0]) > 9 and float(number[0]) % 5 == 0):
                return True

class OCRLabel(OCRBase):
    unsuccessful_return = None
    tp = 'label'
    srcnn_path = 'extracted_data_single/Labels_SRCNN/'

    def image2string(self, img):
        return pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=abcdefghiABCDEFGHI1234 --psm 7')

    def success(self, st):
        whitelist = 'abcdefghiABCDEFGHI1234'
        if len(st) == 1 and st.isalpha() and (st in whitelist):
            return True
        elif len(st) == 2 and st[0].isalpha() and st[1].isdigit() and (st[0] in whitelist):
            return True
        else:
            return False

class BuilderFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, class_name):
        self._builders[key] = class_name

    def create(self, key):
        builder = self._builders.get(key)
        return builder()

factory = BuilderFactory()
factory.register_builder('label', OCRLabel)
factory.register_builder('scale', OCRScale)


def read(tp, path):
    ocr_reader = factory.create(tp)
    result = ocr_reader.recognize_text(path)
    return result
