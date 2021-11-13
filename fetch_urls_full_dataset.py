import os
import re
import hashlib
import requests
import urllib.request
import json

from shutil import copyfile

from figure_separator import extract
from utils import gif_to_jpg, crop_images

OUTPUT_DIR = 'extracted_data_full_dataset'
PATH_extracted = 'extracted_images_gif'

def make_path(path):
    return os.path.join(OUTPUT_DIR, path)

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def make_request(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, os.path.join(make_path(PATH_extracted), "image"))

def read_chunks(input_path, block_size):
    with open(input_path, 'rb') as f_in:
        while True:
            chunk = f_in.read(block_size)
            if chunk:
                yield chunk
            else:
                return

def hash(input_path):
    hf = hashlib.sha1()
    for chunk in read_chunks(input_path, 256 * (128 * hf.block_size)):
        hf.update(chunk)
    return hf.hexdigest()

create_dir(make_path(PATH_extracted))
fnames = []
f = json.load(open('full_dataset.json'))
for entry in f.keys():
    image_info = f[entry]
    doi = image_info['DOI'].replace('/', '%')
    url = image_info['Composite_Figure_URL']
    # TEM_hash = image_info['Hash']
    # fname = doi + '_' + TEM_hash + '.jpg'
    fnames.append(entry)

    make_request(url)
    hash_id = hash(os.path.join(make_path(PATH_extracted), "image"))
    os.rename(os.path.join(make_path(PATH_extracted), "image"), os.path.join(make_path(PATH_extracted), "{}_{}".format(doi, hash_id)))

# Convert from gif to jpg
if not os.path.isdir(make_path("extracted_images_jpg")):
    os.mkdir(make_path("extracted_images_jpg"))
for f in os.listdir(make_path("extracted_images_gif")):
    gif_to_jpg(os.path.join(make_path("extracted_images_gif"), f),
                os.path.join(make_path("extracted_images_jpg"), str(f) + ".jpg"))

# Separate subfigures from composite figures
extract(make_path('extracted_images_jpg'), OUTPUT_DIR)
crop_images(make_path("extracted_images_jpg"), make_path("extracted_subfigures_json"), make_path("extracted_subfigures_png"))

create_dir(make_path('data'))
for i, fname in enumerate(fnames):
    try:
        copyfile(make_path(os.path.join('extracted_subfigures_png', fname)), make_path(os.path.join('data', fname)))
    except:
        print('Could not find ', fname)