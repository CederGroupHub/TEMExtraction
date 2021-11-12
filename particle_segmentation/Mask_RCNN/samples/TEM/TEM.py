"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluation on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from imgaug import augmenters as iaa

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from size_measurement.size_measurement import measure_sizes_single

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# turn off warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical devices: ", physical_devices)
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class TEMConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "TEM"

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # (background + cells 1,2,3)

    BACKBONE = "resnet101"

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    
    RPN_ANCHOR_RATIOS = [0.25, 0.5, 1, 2, 4]

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0, 0, 0])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    IMAGE_CHANNEL_COUNT = 3
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400 

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


############################################################
#  Dataset
############################################################

class TEMDataset(utils.Dataset):
    def load_TEM(self, image_dir, annotations_file):
        """Load a subset of the TEM dataset.
        dataset_dir: The root directory of the COCO dataset.
        annotations_file: JSON file connecting COCO format annotations.
        """
        coco = COCO(annotations_file)

        class_ids = sorted(coco.getCatIds())

        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("TEM", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "TEM", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "TEM":
            return super(TEMDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []

        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "TEM.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TEMDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "TEM":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(TEMDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def visualize_instances(dataset_val, inference_config, gt=False, image_id=None):
    # if image_id is None:
    #     image_id = random.choice(dataset_val.image_ids)

    for i, image_id in enumerate(dataset_val.image_ids):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config, image_id)
    
        if gt == True:
            visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                        dataset_val.class_names, figsize=(8, 8), index=i)
        else:
            results = model.detect([original_image], verbose=1)
            r = results[0]
            visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                        dataset_val.class_names, r['scores'], index=i)


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "TEM"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
    
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

def evaluate_ap(model, dataset, inference_config, coco, limit=0, image_ids=None):
    # Pick COCO images from the dataset
    # image_ids = image_ids or dataset.image_ids
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config, image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        print("AP for {}: {}".format(image_id, AP))
    print("mAP for {} images: {}".format(len(image_ids),np.mean(APs)))


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on TEM Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test' or 'viz' on TEMDataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/images/",
                        help='Directory of the image dataset')
    parser.add_argument('--train_ann', required=False,
                        metavar="training annotations file",
                        help='Path to JSON train annotations file')
    parser.add_argument('--test_ann', required=False,
                        metavar="test annotations file",
                        help='Path to JSON test annotations file')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco' or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image_id', required=False,
                        default=4,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--scales_path', required=False)
    parser.add_argument('--output_dir', required=False)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Train Ann file: ", args.train_ann)
    print("Test Ann file: ", args.test_ann)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TEMConfig()
    else:
        class InferenceConfig(TEMConfig):
            # Set batch size to 1 to run one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            # Don't resize imager for inferencing
            # IMAGE_RESIZE_MODE = "pad64"

            # Non-max suppression threshold to filter RPN proposals.
            # You can increase this during training to generate more propsals.
            RPN_NMS_THRESHOLD = 0.9

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = TEMDataset()
        dataset_train.load_TEM(args.dataset, args.train_ann)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = TEMDataset()
        dataset_val.load_TEM(args.dataset, args.test_ann)
        dataset_val.prepare()

        # Image augmentation
        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=50,
                    augmentation=augmentation,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100,
                    augmentation=augmentation,
                    layers='all')

    elif args.command == "test":
        # Validation dataset
        dataset_val = TEMDataset()
        coco = dataset_val.load_TEM(args.dataset, args.test_ann)
        dataset_val.prepare()
        # evaluate_coco(model, dataset_val, coco, "segm", limit=0)
        evaluate_ap(model, dataset_val, config, coco, limit=0, image_ids=args.image_id)

    elif args.command == 'viz':
        dataset_val = TEMDataset()
        coco = dataset_val.load_TEM(args.dataset, args.test_ann)
        dataset_val.prepare()
        visualize_instances(dataset_val, config, gt=False, image_id=None)
    
    elif args.command == 'infer':
        if not os.path.isdir('object_classes'):
            os.mkdir('object_classes')
        # if not os.path.isdir('masks'):
        #     os.mkdir('masks')
        if os.path.exists('object_classes/class_ids.json'):
            json_anns = json.load(open('object_classes/class_ids.json'))
        else:
            json_anns = {}
        for i, f in enumerate(os.listdir(args.dataset)):
            original_image = np.array(Image.open(os.path.join(args.dataset, f)))
            results = model.detect([original_image], verbose=1)
            r = results[0]
            class_names = ['BG', 'particle', 'rod', 'cube', 'triangle']
            # masks, class_ids = visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
            #                             class_names, r['scores'], index=i, filename=f)
            measure_sizes_single(f, r['masks'], r['class_ids'].tolist(), args.scales_path, args.output_dir)
            # with open("masks/{}.npy".format(f), 'wb') as fname:
            #     np.save(fname, r['masks'])
            json_anns[f] = r['class_ids'].tolist()
            if i % 1 == 0:
                with open('object_classes/class_ids.json', 'w') as fname:
                    json.dump(json_anns, fname)

            # if i == 10:
            #     break
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
