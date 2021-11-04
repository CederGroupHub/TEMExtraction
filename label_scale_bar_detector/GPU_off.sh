cd label_scale_bar_detector/localizer/darknet/

sed -i 's/GPU=1/GPU=0/' Makefile
sed -i 's/CUDNN=1/CUDNN=0/' Makefile
sed -i 's/CUDNN_HALF=1/CUDNN_HALF=0/' Makefile

cd ../..