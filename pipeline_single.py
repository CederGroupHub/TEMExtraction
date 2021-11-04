import os
import re
import time
import torch
import shutil
import subprocess
import pandas as pd

from figure_separator import extract
from ImageSoup import extract_figures_single_paper
from classifier import classify
from torchvision import datasets
from label_scale_bar_detector.localizer import detect, GPU_on, GPU_off
from utils import (image_extractor, gif_to_jpg, crop_images, crop_labels, crop_scales,
                read_OCR_from_folder, run_segmentation, measure_bars, drop_zeros)


def run_pipeline_single(output_dir, gpu, publisher, html_source):

    # Set GPU option in the Makefile of label_scale_bar_detector
    if gpu:
        GPU_on()
    else:
        GPU_off()

    def make_path(path):
        return os.path.join(output_dir, path)

    def create_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    # Extract images from htmls
    print("publisher pipeline", publisher)
    meta = extract_figures_single_paper(publisher, html_source)

    # publishers = ["Elsevier", "Nature Publishing Group", "The Royal Society of Chemistry", "Springer"]
    # # Extract images
    # for publisher in publishers:
    #     image_extractor.extract(publisher, download=True)
    # create_dir(make_path('extracted_images_gif'))
    image_extractor.extract_single_paper(publisher, meta)

    # Convert from gif to jpg
    # if not os.path.isdir(make_path("extracted_images_jpg")):
    #     os.mkdir(make_path("extracted_images_jpg"))
    for f in os.listdir(make_path("extracted_images_gif")):
        create_dir(os.path.join(make_path('extracted_images_jpg')))
        gif_to_jpg(os.path.join(make_path("extracted_images_gif"), f),
                os.path.join(make_path("extracted_images_jpg"), str(f) + ".jpg"))

    # Separate subfigures from composite figures
    extract(make_path('extracted_images_jpg'), output_dir)
    crop_images(make_path("extracted_images_jpg"), make_path("extracted_subfigures_json"), make_path("extracted_subfigures_png"))

    classify(make_path('extracted_subfigures_png'), 'sem_tem_other', output_dir, gpu)

    # # Classify subfigures into TEM, XRD, Other
    # classify(make_path('extracted_subfigures_png'), 'tem_xrd_other', output_dir, GPU)

    # # Classify TEM images into subcategories
    # classify(make_path('TEM'), 'tem_subcategories', output_dir, GPU)

    # Classify TEM images into Particulate and Non-particulate
    classify(make_path('TEM'), 'particulate', output_dir, gpu)

    # Detect labels, scales and bars
    detect(make_path('Particulate'))

    # Crop located labels, scales and bars
    crop_scales(make_path('Particulate'), 'label_scale_bar_detector/localizer/darknet/result.json', output_dir)

    # Read labels and scales
    labels_path = make_path("label")
    scales_path = make_path("scale")
    read_OCR_from_folder('label', labels_path, output_dir)
    read_OCR_from_folder('scale', scales_path, output_dir)
    drop_zeros(make_path("scales.csv"))

    # measure bar lengths and add to csv
    measure_bars(make_path("bar"), make_path("scales.csv"), scales_path)

    # Run particle segmentation
    run_segmentation(make_path("Particulate"), "particle_segmentation/Mask_RCNN", make_path("scales.csv"), output_dir)
