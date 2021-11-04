import os
import shutil
import subprocess


def GPU_on():
    subprocess.check_call('bash label_scale_bar_detector/GPU_on.sh', shell=True)

def GPU_off():
    subprocess.check_call('bash label_scale_bar_detector/GPU_off.sh', shell=True)

def detect(data_dir):
    try:
        os.remove('label_scale_bar_detector/localizer/darknet/data/test.txt')
    except:
        print("File test.txt doesn't exist. Nothing to delete!")
    try:
        shutil.rmtree('label_scale_bar_detector/localizer/darknet/data/test')
    except:
        print("Directory test doesn't exist. Nothing to delete!")
    os.mkdir("label_scale_bar_detector/localizer/darknet/data/test")
    with open('label_scale_bar_detector/localizer/darknet/data/test.txt', 'w') as outfile:
        for f in os.listdir(data_dir):
            outfile.write('data/test/' + str(f) + '\n')
    for f in os.listdir(data_dir):
        shutil.copyfile(os.path.join(data_dir, str(f)), 'label_scale_bar_detector/localizer/darknet/data/test/' + str(f))
    subprocess.check_call('bash label_scale_bar_detector/localize.sh', shell=True)
