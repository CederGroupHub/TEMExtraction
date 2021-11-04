import os
import io
import re
import hashlib
import mimetypes
import subprocess
import requests
import urllib.request
import json
from PIL import Image, ImageDraw
from label_scale_bar_detector.OCR import read
import pandas as pd


class ImageExtractor(object):
    PATH_extracted = "extracted_data_single/extracted_images_gif"
    if not os.path.isdir(PATH_extracted):
        os.makedirs(PATH_extracted)

    def read_chunks(self, input_path, block_size):
        """Iterate over ``block_size`` chunks of file at ``input_path``.

        :param str input_path: the path to the input file to iterate over.
        :param int block_size: the size of the chunks to return at each
        iteration.

        :yields: a binary chunk of the file at ``input_path`` of size
        ``block_size``.
        """
        with open(input_path, 'rb') as f_in:
            while True:
                chunk = f_in.read(block_size)
                if chunk:
                    yield chunk
                else:
                    return

    def hash(self, input_path):
        hf = hashlib.sha1()
        for chunk in self.read_chunks(input_path, 256 * (128 * hf.block_size)):
            hf.update(chunk)
        return hf.hexdigest()

    def retrieve_image(self, id):
        import gridfs
        if not gridfs.exists(id):
            raise Exception("mongo file does not exist! {0}".format(id))
        im_stream = gridfs.get(id)
        im = Image.open(im_stream)
        im.show()
        return im

    def is_single(self, caption):
        if re.search(r'\([a-zA-Z][\d]*\)', caption) == None and \
            re.search(r'^[a-zA-Z][\d]*,\s', caption) == None and \
            re.search(r'\s[a-zA-Z][\d]*,\s', caption) == None and \
            re.search(r'\([a-zA-Z][\d]*,[a-zA-Z][\d]*\)', caption) == None and \
            re.search(r'^[a-zA-Z][\d]*\)\s', caption) == None and \
            re.search(r'\s[a-zA-Z][\d]*\)\s', caption) == None and \
            re.search(r'\s[bB]\s', caption) == None and \
            re.search(r'\([a-zA-Z][\d]*\-[a-zA-Z][\d]*\)', caption) == None:
            return True
        else:
            return False

    def is_TEM_XRD(self, caption: str):
        is_gold = False
        is_single = self.is_single(caption)

        is_shape = False
        if re.search(r'Au', caption) != None or re.search(r'Gold', caption) != None or re.search(r'gold', caption) != None:
            is_gold = True

        if re.search(r'cube', caption.lower()) != None or re.search(r'sphere', caption.lower()) != None or re.search(r'nano\s*rod', caption.lower()) != None or re.search(r'[\.\s\)]rod', caption.lower()) != None or \
            re.search(r'triang', caption.lower()) != None or re.search(r'prism', caption.lower()) != None or re.search(r'au\s*n[r]', caption.lower()) != None or re.search(r'au\s*n[r]', caption.lower()) != None:
            is_shape = True

        if (re.search(r'[\.\s\)][ST]EM', caption) != None or re.search(r'^[ST]EM', caption) != None or "transmission electron" in caption.lower() or "scanning electron" in caption.lower()) and is_gold and is_shape:
            return True
        else:
            return False

    def make_request(self, url):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, os.path.join(self.PATH_extracted, "image"))

    def extract_single_paper(self, publisher: str, meta: dict):
        if not os.path.isdir(self.PATH_extracted):
            os.mkdir(self.PATH_extracted)

        figures_modified = meta['Figures'].copy()
        # figures_modified = doc["Figures"].copy()
        paper_ctr = 0
        if 'Figures' not in meta:
            return False
        for i, image in enumerate(meta['Figures']):
            small = False
            if not self.is_TEM_XRD(image["Caption"]):
                continue
            url = image["Image_URL"]
            if publisher == "Elsevier":
                if url.endswith('.sml'):
                    url = url.replace('.sml', '_lrg.jpg')
                elif url.endswith('_lrg.jpg'):
                    url = url
                elif url.endswith('.jpg'):
                    url = url.replace('.jpg', '_lrg.jpg')
                # Try downloading large sized image.
                try:
                    self.make_request(url)
                except:
                    url = url.replace('_lrg.jpg', '.jpg')
                    # Try downloading medium sized image.
                    try:
                        self.make_request(url)
                    except:
                        print("Unsuccessful ", image["Image_URL"])
                        continue
                    # except:
                    #     # Try downloading small sized image.
                    #     try:
                    #         self.make_request(image["Image_URL"])
                    #         small = True
                    #     except:
                    #         print("Unsuccessful ", image["Image_URL"])
                    #         continue
                hash_id = self.hash(os.path.join(self.PATH_extracted, "image"))
                figures_modified[i]["Hash"] = hash_id
                figures_modified[i]["Download_URL"] = url
                figures_modified[i]["Shape_filter"] = True
                meta['Figures'] = figures_modified
                os.rename(os.path.join(self.PATH_extracted, "image"), os.path.join(self.PATH_extracted, "{}".format(hash_id)))
            else:
                try:
                    self.make_request(url)
                except:
                    if publisher == "Nature Publishing Group":
                        try:
                            print("Trying year correction")
                            year = re.search(r'.*(201.).*', url).group(1)
                            new_url = url.replace(year, str(int(year)-1))
                            self.make_request(new_url)
                            url = new_url
                        except:
                            try:
                                print("Trying extension correction")
                                new_url = url.replace('.jpg', '.png')
                                self.make_request(new_url)
                                url = new_url
                            except:
                                try:
                                    new_url = url.replace(".jpg", ".gif")
                                    self.make_request(new_url)
                                    url = new_url
                                except:
                                    print("Unsuccessful even after year and extension modification ", image["Image_URL"])
                                    continue
                    else:
                        print("Unsuccessful ", url)
                        continue
                hash_id = self.hash(os.path.join(self.PATH_extracted, "image"))
                figures_modified[i]["Hash"] = hash_id
                figures_modified[i]['Download_URL'] = url
                figures_modified[i]['Shape_filter'] = True
                meta['Figures'] = figures_modified
                os.rename(os.path.join(self.PATH_extracted, "image"), os.path.join(self.PATH_extracted, "{}".format(hash_id)))
        return meta

image_extractor = ImageExtractor()

def crop_labels(path_img, path_ann, path_save):
    annotations = json.load(open(path_ann))
    for ann in annotations:
        name = ann['filename'].split('/')[2]
        objects = ann['objects']
        if len(objects) == 0:
            continue
        elif len(objects) == 1:
            coords = objects[0]['relative_coordinates']
            confidence = objects[0]['confidence']
        else:
            confidence = 0
            highest_confidence_index = 0
            for i, obj in enumerate(ann['objects']):
                confidence_new = obj['confidence']
                if confidence_new > confidence:
                    highest_confidence_index = i
            coords = objects[highest_confidence_index]['relative_coordinates']
            confidence = objects[highest_confidence_index]['confidence']
        # Crop out bbox and save in path_save
        img = Image.open(os.path.join(path_img, name))
        w, h = img.size
        left = (coords['center_x'] - coords['width']/2) * w
        right = (coords['center_x'] + coords['width']/2) * w
        top = (coords['center_y'] - coords['height']/2) * h
        bottom = (coords['center_y'] + coords['height']/2) * h

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(os.path.join(path_save, 'label_' + str(name)))

def crop_scales(path_img, path_ann, save_dir):
    annotations = json.load(open(path_ann))
    types = ['bar', 'scale', 'label']

    for ann in annotations:
        name = ann['filename'].split('/')[2]
        for tp in types:
            output_dir = os.path.join(save_dir, tp)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            objects = [obj for obj in ann['objects'] if obj["name"] == tp]
            if len(objects) == 0:
                continue
            elif len(objects) == 1:
                coords = objects[0]['relative_coordinates']
                confidence = objects[0]['confidence']
            else:
                confidence = 0
                highest_confidence_index = 0
                for i, obj in enumerate(objects):
                    confidence_new = obj['confidence']
                    if confidence_new > confidence:
                        highest_confidence_index = i
                coords = objects[highest_confidence_index]['relative_coordinates']
                confidence = objects[highest_confidence_index]['confidence']
            # Crop out bbox and save in path_save
            img = Image.open(os.path.join(path_img, name))
            w, h = img.size
            left = (coords['center_x'] - coords['width']/2) * w
            right = (coords['center_x'] + coords['width']/2) * w
            top = (coords['center_y'] - coords['height']/2) * h
            bottom = (coords['center_y'] + coords['height']/2) * h

            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(os.path.join(output_dir, str(tp) + '_' + str(name)))

def crop_images(path_img, path_ann, path_save):
    if not os.path.isdir(path_save):
        os.mkdir(path_save)
    for f in os.listdir(path_img):
        img = Image.open(os.path.join(path_img, f))

        annotations = json.load(open(os.path.join(path_ann, f.replace(".jpg", ".jpg.json"))))
        w, h = img.size
        print(w, h)
        margin = 0
        for i, im in enumerate(annotations):
            left = im['x']
            if left - margin >= 0:
                left -= margin

            right = im['x'] + im['w']
            if right + margin <= w:
                right += margin

            top = im['y']
            if top - margin >= 0:
                top -= margin

            bottom = im['y'] + im['h']
            if bottom + margin <= h:
                bottom += margin

            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(os.path.join(path_save, f.replace("jpg", "") + "{}.jpg".format(i)))


def gif_to_jpg(src_path, dest_path):
    try:
        img = Image.open(src_path)
        img.convert('RGB').save(dest_path)
    except:
        print("couldn't be converted", src_path)

def read_OCR_from_folder(tp, src_path, dest_path):
    # Run OCR on the extracted labels
    if tp == 'scale':
        columns = ['filename', 'digit', 'unit']
        fname = "scales.csv"
        if not os.path.isdir(os.path.join(dest_path, 'Scales_bicubic')):
            os.mkdir(os.path.join(dest_path, 'Scales_bicubic'))
        if not os.path.isdir(os.path.join(dest_path, 'Scales_SRCNN')):
            os.mkdir(os.path.join(dest_path, 'Scales_SRCNN'))
    elif tp == 'label':
        columns = ['filename', 'label']
        fname = "labels.csv"
        if not os.path.isdir(os.path.join(dest_path, 'Labels_bicubic')):
            os.mkdir(os.path.join(dest_path, 'Labels_bicubic'))
        if not os.path.isdir(os.path.join(dest_path, 'Labels_SRCNN')):
            os.mkdir(os.path.join(dest_path, 'Labels_SRCNN'))

    ctr_total = 0
    ctr = 0
    data = []
    for f in os.listdir(src_path):
        ctr_total += 1
        # if ctr_total == 6:
        #     break
        f_new = f.replace("scale_", "").replace("label_", "")
        if tp == 'label':
            text = read(tp, os.path.join(src_path, f))
            if text:
                data.append([f_new, text])
                ctr += 1
        elif tp == 'scale':
            digit, unit = read(tp, os.path.join(src_path, f))
            if digit and unit:
                data.append([f_new, digit, unit])
                ctr += 1

        if ctr != 0 and ctr % 10 == 0:
            print("Labels extracted:", ctr)
            print("Labels seen:", ctr_total)
            print("Extraction rate:", float(ctr) / ctr_total * 100, '%')

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(dest_path, fname), index=False)

def run_segmentation(images_dir, mask_rcnn_dir, scales_path, output_dir):
    os.chdir(mask_rcnn_dir)
    user = os.getenv("USER")
    subprocess.check_call('python samples/TEM/TEM.py infer'
                            ' --model {}'
                            ' --dataset {}'
                            ' --scales_path {}'
                            ' --output_dir {}'.format(
                                './logs/tem/mask_rcnn_tem_0200.h5',
                                os.path.join("/home/{}/TEM-XRD-pipeline/".format(user), images_dir),
                                scales_path,
                                output_dir),
                                shell=True
                        )
    os.chdir('/home/{}/TEM-XRD-pipeline'.format(user))

def drop_zeros(scales_csv_path):
    df = pd.read_csv(scales_csv_path)
    df.drop(df[df.digit == 0].index, inplace=True)
    df.to_csv(scales_csv_path, index=False)

def measure_bars(bars_dir, scales_csv_path, scales_dir):
    df  = pd.read_csv(scales_csv_path)
    bars = os.listdir(bars_dir)
    scales = os.listdir(scales_dir)
    scales = [i.replace('scale_', '') for i in scales]

    data = df.to_dict('records')
    columns = list(df.columns)
    if 'bar_length' not in columns:
        columns.append('bar_length')

    for fname in bars:
        im = Image.open(os.path.join(bars_dir, fname))
        width, _ = im.size
        f = fname.replace("bar_", "")

        found = False
        for i, el in enumerate(data):
            if el['filename'] == f:
                data[i]['bar_length'] = width
                found = True
                break

        if found == False:
            data.append({'filename': f, 'digit': 'None', 'unit': 'None', 'bar_length': width})

    for i, el in enumerate(data):
        if 'bar_length' not in el:
            if data[i]['filename'] in scales:
                im = Image.open(os.path.join(scales_dir, 'scale_' + data[i]['filename']))
                width, _ = im.size
                data[i]['bar_length'] = width
            else:
                data[i]['bar_length'] = 'None'

    df_out = pd.DataFrame(data, columns=columns)
    df_out.to_csv(scales_csv_path, index=False)

