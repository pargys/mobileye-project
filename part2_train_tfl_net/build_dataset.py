import numpy as np
from part1_light_spots_detection.run_attention import find_img2d_candidates
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import random


def remove_candidate_by_index(candidates_x, candidates_y, index):
    candidates_x.pop(index)
    candidates_y.pop(index)


def find_tfl_coords(image, label):
    image = np.array(image)
    candidates_x, candidates_y = find_img2d_candidates(image)
    tfl_x = []
    tfl_y = []
    not_tfl_x = []
    not_tfl_y = []
    num_of_balance_crop_imgs = 5

    while len(tfl_x) < num_of_balance_crop_imgs and len(tfl_y) < num_of_balance_crop_imgs and len(candidates_x):
        index = random.randint(0, len(candidates_x)-1)

        if len(tfl_x) < num_of_balance_crop_imgs and 19 == label[candidates_x[index]][candidates_y[index]]:
            tfl_x.append(candidates_x[index])
            tfl_y.append(candidates_y[index])

        elif len(not_tfl_x) < num_of_balance_crop_imgs:
            not_tfl_x.append(candidates_x[index])
            not_tfl_y.append(candidates_y[index])

        remove_candidate_by_index(candidates_x, candidates_y, index)

    min_coords = min(len(tfl_x), len(not_tfl_x))
    crop_x_imgs = tfl_x[:min_coords] + not_tfl_x[:min_coords]
    crop_y_imgs = tfl_y[:min_coords] + not_tfl_y[:min_coords]
    labels = [1 for i in range(min_coords)] + [0 for i in range(min_coords)]

    return crop_x_imgs, crop_y_imgs, labels


def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0


def crop(image, coords_x, coords_y):
    image = np.pad(image, 40, pad_with_zeros)[:, :, 40:43]
    crop_imgs = []
    for x, y in zip(coords_x, coords_y):
        crop_image = image[x:x+81, y:y+81]
        crop_imgs.append(crop_image)
    return np.array(crop_imgs)


def insert_bin_file(data_name, crop_imgs, labels):
    data_root_path = '../correct_data/' + data_name
    with open(f"{data_root_path}/data.bin", "ab") as data_file:
        for image in crop_imgs:
            np.array(image, dtype=np.uint8).tofile(data_file)
    with open(f"{data_root_path}/labels.bin", "ab") as labels_file:
        for label in labels:
            labels_file.write((label).to_bytes(1, byteorder='big', signed=False))


def create_bin_file(data_name):
    path_imgs = '../data/leftImg8bit'
    path_labels = '../data/gtFine'

    for subdir, dirs, files in os.walk(path_imgs + '/' + data_name):
        for dir in dirs:
            i = 0
            print('\n', dir)
            imgs_list = glob.glob(os.path.join(path_imgs + '/' + data_name + '/' + dir, '*_leftImg8bit.png'))
            labels_list = glob.glob(os.path.join(path_labels + '/' + data_name + '/' + dir, '*_gtFine_labelIds.png'))
            for img_path, label_path in zip(imgs_list, labels_list):
                i += 1
                image = Image.open(img_path)
                label = np.array(Image.open(label_path))

                coords_x, coords_y, labels = find_tfl_coords(image.convert('L'), label)
                crop_imgs = crop(np.array(image), coords_x, coords_y)

                insert_bin_file(data_name, crop_imgs, labels)
                if i % 5 == 0:
                    print(i, end=" ")


def data_set():
    create_bin_file('train')
    create_bin_file('val')


def main():
    data_set()
    plt.show(block=True)


if __name__ == '__main__':
    main()