import os
import math
import json
import numpy as np
import pandas as pd
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def centre_of_mass(img_np):
    x = 0.0
    y = 0.0
    total = 0.0
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            greater_than_zero = int(img_np[i][j] > 0)
            x += i * greater_than_zero
            y += j * greater_than_zero
            total += greater_than_zero
    return (int(x / total), int(y / total))

def get_contours(mask):
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    return contours

def points_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def measure_rod_size(mask):
    centre_x, centre_y = centre_of_mass(mask)
    # _, ax = plt.subplots(1, figsize=(12, 12))

    # plt.imshow(mask)
    # plt.show()

    contours = get_contours(mask)
    if len(contours) == 0:
        return None, None
    distances = []
    for point in contours[0]:
        x1 = point[0]
        y1 = point[1]
        distances.append((x1, y1, points_distance(x1, y1, centre_x, centre_y)))
    #     plt.plot([y1, centre_y], [x1, centre_x], marker='o')
    # plt.imshow(mask)
    # plt.show()
    distances.sort(key=lambda x: x[2], reverse=True)
    # plt.plot([distances[0][1], centre_y], [distances[0][0], centre_x], marker='o')
    # plt.show()
    # plt.imshow(mask)
    # plt.plot([distances[1][1], centre_y], [distances[1][0], centre_x], marker='o')
    # plt.show()
    # try:
    # mean_length = (distances[0] + distances[1]) / 2
    # except:
        # return None
    return (distances[0][2] * 2, distances[-1][2] * 2)

def measure_cube_size(mask):
    centre_x, centre_y = centre_of_mass(mask)
    # _, ax = plt.subplots(1, figsize=(12, 12))

    # plt.imshow(mask)
    # plt.show()

    contours = get_contours(mask)
    if len(contours) == 0:
        return None
    distances = []
    for point in contours[0]:
        x1 = point[0]
        y1 = point[1]
        distances.append((x1, y1, points_distance(x1, y1, centre_x, centre_y)))
    #     plt.plot([y1, centre_y], [x1, centre_x], marker='o')
    # plt.imshow(mask)
    # plt.show()
    distances.sort(key=lambda x: x[2])
    # plt.plot([distances[0][1], centre_y], [distances[0][0], centre_x], marker='o')
    # plt.show()
    # try:
    # mean_centre_to_side = (distances[0] + distances[1] + distances[2] + distances[3]) / 4
    # except:
        # return None
    return distances[0][2] * 2

def measure_triangle_size(mask):
    centre_x, centre_y = centre_of_mass(mask)
    # _, ax = plt.subplots(1, figsize=(12, 12))

    # plt.imshow(mask)
    # plt.show()

    contours = get_contours(mask)
    if len(contours) == 0:
        return None
    distances = []
    for point in contours[0]:
        x1 = point[0]
        y1 = point[1]
        # distances.append(points_distance(x1, y1, centre_x, centre_y))
        distances.append((x1, y1, points_distance(x1, y1, centre_x, centre_y)))
    #     plt.plot([y1, centre_y], [x1, centre_x], marker='o')
    plt.imshow(mask)
    # plt.show()
    distances.sort(key=lambda x: x[2], reverse=True)
    # plt.plot([distances[-1][1], centre_y], [distances[-1][0], centre_x], marker='o')
    # plt.show()
    #     plt.plot([y1, centre_y], [x1, centre_x], marker='o')
    # plt.imshow(mask)
    # plt.show()
    # distances.sort(reverse=True)
    # try:
    # mean_centre_to_vertex = (distances[0] + distances[1] + distances[2]) / 3
    # mean_centre_to_side = (distances[-1] + distances[-2] + distances[-3]) / 3
    # except:
        # return None

    return distances[0][2] + distances[-1][2]

def measure_particle_size(mask):
    centre_x, centre_y = centre_of_mass(mask)
    # _, ax = plt.subplots(1, figsize=(12, 12))

    # Visualize the centres to verify that they are correct
    for point_x in range(1):
        for point_y in range(1):
            mask[centre_x + point_x][centre_y + point_y] = 0
    # plt.imshow(mask)
    # plt.show()

    contours = get_contours(mask)
    if len(contours) == 0:
        return None
    # for verts in contours:
    #     # Subtract the padding and flip (y, x) to (x, y)
    #     verts = np.fliplr(verts) - 1
    #     p = Polygon(verts, facecolor="none")
    #     ax.add_patch(p)

    distances = []
    for point in contours[0]:
        x1 = point[0]
        y1 = point[1]
        distances.append(points_distance(x1, y1, centre_x, centre_y))
    #     plt.plot([y1, centre_y], [x1, centre_x], marker='o')
    # plt.imshow(mask)
    # plt.show()
    return sum(distances)/len(distances) * 2

def plot_shape(sizes_list, unit, num_plots):
    fig, ax = plt.subplots(num_plots)
    fig.suptitle('Particle Size Distribution')
    current_plot = 0

    def plot(x, xlabel, title, ax):
        ax.hist(x, density=True, bins=30)
        ax.set(xlabel=xlabel, ylabel="No. of particles")
        ax.set_title(title)
        # plt.hist(x, density=True, bins=30)  # density=False would make counts
        # plt.ylabel("No. of particles")
        # plt.xlabel(xlabel)
        # plt.title(title)
        # plt.show()

    if len(sizes_list["sphere"]["diameter"]) > 0:
        if num_plots == 1:
            plot(sizes_list["sphere"]["diameter"], "Diameter ({})".format(unit), "Sphere", ax)
        else:
            plot(sizes_list["sphere"]["diameter"], "Diameter ({})".format(unit), "Sphere", ax[current_plot])
        current_plot += 1
    if len(sizes_list["rod"]["length"]) > 0:
        if num_plots == 1:
            plot(sizes_list["rod"]["length"], "Length ({})".format(unit), "Rod (Length)", ax)
        else:
            plot(sizes_list["rod"]["length"], "Length ({})".format(unit), "Rod (Length)", ax[current_plot])
        current_plot += 1
    if len(sizes_list["rod"]["width"]) > 0:
        if num_plots == 1:
            plot(sizes_list["rod"]["width"], "Width ({})".format(unit), "Rod (Width)", ax)
        else:
            plot(sizes_list["rod"]["width"], "Width ({})".format(unit), "Rod (Width)", ax[current_plot])
        current_plot += 1
    if len(sizes_list["cube"]["side"]) > 0:
        if num_plots == 1:
            plot(sizes_list["cube"]["side"], "Side length ({})".format(unit), "Cube", ax)
        else:
            plot(sizes_list["cube"]["side"], "Side length ({})".format(unit), "Cube", ax[current_plot])
        current_plot += 1
    if len(sizes_list["triangle"]["height"]) > 0:
        if num_plots == 1:
            plot(sizes_list["triangle"]["height"], "Height ({})".format(unit), "Triangle", ax)
        else:
            plot(sizes_list["triangle"]["height"], "Height ({})".format(unit), "Triangle", ax[current_plot])
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plot/plot.jpg', bbox_inches='tight', pad_inches=0)

def measure_sizes_single(fname, masks, class_ids, scales_csv_path, output_dir):
    user = os.getenv("USER")
    os.chdir('/home/{}/AuSEM'.format(user))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    out_path = os.path.join(output_dir, "sizes.json")
    if os.path.exists(out_path):
        sizes_json = json.load(open(out_path))
    else:
        sizes_json = {}

    if scales_csv_path != '':
        df = pd.read_csv(scales_csv_path)
    
    try:
        digit = df.loc[df.filename == fname, "digit"].iloc[0]
        unit = df.loc[df.filename == fname, "unit"].iloc[0]
        bar_width = df.loc[df.filename == fname, "bar_length"].iloc[0]
    except:
        bar_width = "None"
        digit = "None"

    if bar_width == "None" or digit == "None":
        conversion_factor = 1
        unit = "pixels"
    else:
        conversion_factor = float(digit) / float(bar_width)

    sizes_list = {
        'rod': {
            'length': [],
            'width': [],
        },
        'sphere': {
            'diameter': []
        },
        'triangle': {
            'height': []
        },
        'cube': {
            'side': []
        }
    }

    for i, class_id in enumerate(class_ids):
        if (masks[:, :, i] > 0).sum() == 0:
            print("skipping")
            continue
        if class_id == 1:
            size = measure_particle_size(masks[:, :, i])
            if size != None:
                sizes_list['sphere']['diameter'].append(size * conversion_factor)
                # shapes.append('sphere')
        elif class_id == 2:
            length, width = measure_rod_size(masks[:, :, i])
            if length != None or width != None:
                sizes_list['rod']['length'].append(length * conversion_factor)
                sizes_list['rod']['width'].append(width * conversion_factor)
                # shapes.append('rod_length')
                # shapes.append('rod_width')
        elif class_id == 3:
            size = measure_cube_size(masks[:, :, i])
            if size != None:
                sizes_list['cube']['side'].append(size * conversion_factor)
                # shapes.append('cube')
        elif class_id == 4:
            size = measure_triangle_size(masks[:, :, i])
            if size != None:
                sizes_list['triangle']['height'].append(size * conversion_factor)
                # shapes.append('triangle')
        # sizes_list.append(size)
        # print(sizes_list)
    sizes_json[fname] = {}
    sizes_json[fname]["Size"] = sizes_list
    sizes_json[fname]["Unit"] = unit
    with open(out_path, 'w') as outfile:
        json.dump(sizes_json, outfile)
    # if i % 20 == 0 and i != 0:
    #     print(i, " images completed!")
    os.chdir('/home/{}/AuSEM/particle_segmentation/Mask_RCNN'.format(user))

def main(masks_dir, class_ids_path, scales_csv_path, output_dir):
    class_ids_all = json.load(open(class_ids_path))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    out_path = os.path.join(output_dir, "sizes.json")
    if os.path.exists(out_path):
        sizes_json = json.load(open(out_path))
    else:
        sizes_json = {}

    if scales_csv_path != '':
        df = pd.read_csv(scales_csv_path)

    # if not os.path.isdir('./plot'):
    #     os.mkdir('./plot')

    # num_plots = 0
    shapes = []
    for i, f in enumerate(os.listdir(masks_dir)):
        masks = np.load(os.path.join(masks_dir, f))
        class_ids = class_ids_all[f.replace(".npy", "")]

        try:
            digit = df.loc[df.filename == f.replace(".npy", ""), "digit"].iloc[0]
            unit = df.loc[df.filename == f.replace(".npy", ""), "unit"].iloc[0]
            bar_width = df.loc[df.filename == f.replace(".npy", ""), "bar_length"].iloc[0]
        except:
            bar_width = "None"
            digit = "None"
        if bar_width == "None" or digit == "None":
            conversion_factor = 1
            unit = "pixels"
        else:
            conversion_factor = float(digit) / float(bar_width)
        sizes_list = {
            'rod': {
                'length': [],
                'width': [],
            },
            'sphere': {
                'diameter': []
            },
            'triangle': {
                'height': []
            },
            'cube': {
                'side': []
            }
        }
        for i, class_id in enumerate(class_ids):
            if (masks[:, :, i] > 0).sum() == 0:
                print("skipping")
                continue
            if class_id == 1:
                size = measure_particle_size(masks[:, :, i])
                if size != None:
                    sizes_list['sphere']['diameter'].append(size * conversion_factor)
                    shapes.append('sphere')
            elif class_id == 2:
                length, width = measure_rod_size(masks[:, :, i])
                if length != None or width != None:
                    sizes_list['rod']['length'].append(length * conversion_factor)
                    sizes_list['rod']['width'].append(width * conversion_factor)
                    shapes.append('rod_length')
                    shapes.append('rod_width')
            elif class_id == 3:
                size = measure_cube_size(masks[:, :, i])
                if size != None:
                    sizes_list['cube']['side'].append(size * conversion_factor)
                    shapes.append('cube')
            elif class_id == 4:
                size = measure_triangle_size(masks[:, :, i])
                if size != None:
                    sizes_list['triangle']['height'].append(size * conversion_factor)
                    shapes.append('triangle')
            # sizes_list.append(size)
            # print(sizes_list)
        sizes_json[f.replace(".npy", "")] = {}
        sizes_json[f.replace(".npy", "")]["Size"] = sizes_list
        sizes_json[f.replace(".npy", "")]["Unit"] = unit
        if i % 1 == 0 and i != 0:
            with open(out_path, 'w') as outfile:
                json.dump(sizes_json, outfile)
        if i % 20 == 0 and i != 0:
            print(i, " images completed!")
        # num_plots += len(list(set(shapes)))
        # if unit:
        #     plot_shape(sizes_list, unit, num_plots)
        # else:
        #     plot_shape(sizes_list, "pixels", num_plots)

# if __name__ == '__main__':
#     main("../../particle_segmentation/Mask_RCNN/masks",
#          "../../particle_segmentation/Mask_RCNN/object_classes/class_ids.json")
