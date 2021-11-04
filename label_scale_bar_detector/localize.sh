#!/bin/bash

# Build darknet according to Makefile (as of now will run on cpu)
cd label_scale_bar_detector/localizer/darknet/
make

# need to set our custom cfg to test mode
cd cfg
sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
cd ..

# Remove result.json from previous run
rm -rf result.json
# Run inference on images and save results as json
./darknet detector test data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_best.weights -thresh 0.3 -i 0 -ext_output -dont_show -out result.json < data/test.txt

# Move back to original directory
cd ../../../..
