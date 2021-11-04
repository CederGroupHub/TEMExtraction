import os
import re
import hashlib
import requests
import urllib.request
import json

from shutil import copyfile

from figure_separator import extract
from utils import gif_to_jpg, crop_images


OUTPUT_DIR = "extracted_data_segmentation"

def make_path(path):
    return os.path.join(OUTPUT_DIR, path)

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

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

def make_request(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, os.path.join(PATH_extracted, "image"))

PATH_extracted = make_path("extracted_images_gif")
create_dir(PATH_extracted)

urls = json.load(open(make_path('urls_dois.json')))
for url in urls.keys():
    doi = urls[url]
    doi = doi.replace('/', '%') # filenames cannot have slash
    make_request(url)
    hash_id = hash(os.path.join(PATH_extracted, "image"))
    os.rename(os.path.join(PATH_extracted, "image"), os.path.join(PATH_extracted, "{}_{}".format(doi, hash_id)))

# Convert from gif to jpg
if not os.path.isdir(make_path("extracted_images_jpg")):
    os.mkdir(make_path("extracted_images_jpg"))
for f in os.listdir(make_path("extracted_images_gif")):
    gif_to_jpg(os.path.join(make_path("extracted_images_gif"), f),
                os.path.join(make_path("extracted_images_jpg"), str(f) + ".jpg"))

# Separate subfigures from composite figures
extract(make_path('extracted_images_jpg'), OUTPUT_DIR)
crop_images(make_path("extracted_images_jpg"), make_path("extracted_subfigures_json"), make_path("extracted_subfigures_png"))

filenames = open(make_path('filenames.txt'), 'r')
for line in filenames:
    fname = line.strip()
    copyfile(make_path(os.path.join('extracted_subfigures_png', fname)), make_path(os.path.join('data', fname)))
