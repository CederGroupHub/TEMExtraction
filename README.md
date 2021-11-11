# Setup Instructions

1) Download [SEM TEM Other Classifier weights](https://drive.google.com/file/d/1WcFTsYAxBUp5jic2rsLEHb6fW7Pw9PEn/view?usp=sharing) and place it in `classifier/SEM_TEM_Other_weights`.
3) Download [Particulate Non-Particulate Classifier weights](https://drive.google.com/file/d/1buy3jZ_sVOo9absjWqk8MYwhjFk6JNJC/view?usp=sharing) and place it in `classifier/Particulate_nonParticulate_weights`.
3) Download [Figure separation weights](https://drive.google.com/file/d/18moIauxgQR2-b4XRF7MAzseyq_8WGJ9D/view?usp=sharing) and place it in `figure-separator/data`.
4) Download [SRCNN weights](https://drive.google.com/file/d/1zmBxzC9SVJm9vciOPLbKzVIVlH09UZtW/view?usp=sharing) and place it in `label_scale_bar_detector/OCR/SRCNN-pytorch/weights/`.
5) Download [Darknet weights](https://drive.google.com/file/d/1CR0chidAN8x7LLWcLHYz4QR7pHfsQB8-/view?usp=sharing) and place it in `label_scale_bar_detector/localizer/darknet/backup`.
6) Download [Mask RCNN weights](https://drive.google.com/file/d/1JwXbYScYNlds5g5JHFwEZLwI5eShyaj4/view?usp=sharing) and place it in `particle_segmentation/Mask_RCNN/logs/tem`.

# Installation

Run `conda env create -f environment.yml`.

# Running the pipeline

Run `python pipeline_single.py`.

# Acknowledgements

1) [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
2) [https://github.com/apple2373/figure-separator](https://github.com/apple2373/figure-separator)
3) [https://github.com/yjn870/SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch)
