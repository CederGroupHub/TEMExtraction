import os
import subprocess

def create_output_directories(output_dir):
    output_images_dir = os.path.join(output_dir, 'annotated_subfigures_png')
    output_json_dir = os.path.join(output_dir, 'extracted_subfigures_json')

    if not os.path.isdir(output_images_dir):
        os.mkdir(output_images_dir)
    if not os.path.isdir(output_json_dir):
        os.mkdir(output_json_dir)
    return output_images_dir, output_json_dir

def extract(images_dir, output_dir, weights_dir='figure_separator/data/figure-sepration-model-submitted-544.pb', annotate=1):
    output_images_dir, output_json_dir = create_output_directories(output_dir)

    subprocess.check_call('python figure_separator/main.py'
                            ' --model {}'
                            ' --images {}'
                            ' --annotate {}'
                            ' --output_images {}'
                            ' --output_json {}'.format(
                                weights_dir,
                                images_dir,
                                annotate,
                                output_images_dir,
                                output_json_dir),
                                shell=True
                            )
