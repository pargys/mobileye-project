import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def insert_bin_file(type, image):
    data_root_path = '../../' + type + '/train'
    with open(f"{data_root_path}/data.bin", "ab") as data_file:
        np.array(image, dtype=np.uint8).tofile(data_file)


def flip(im):
    new = Image.new("RGB", (81, 81))
    new_image_list = []

    for x in range(81):
        temp = []
        for y in range(81):
            pixel = im[x][y]
            new_pixel = (int(pixel[0]),
                         int(pixel[1]),
                         int(pixel[2]))
            temp.append(new_pixel)

        for i in temp[::-1]:
            new_image_list.append(i)
    new.putdata(new_image_list)
    new.save("mirror.png")
    insert_bin_file('my_flip_big_data', new)


def change_brightness(original_image, action, extent):
    new_image = Image.new('RGB', (81, 81))
    brightness_multiplier = 1.0
    new_image_list = []

    if action == 'lighten':
        brightness_multiplier += (extent / 100)
    else:
        brightness_multiplier -= (extent / 100)

    for i in range(81):
        for j in range(81):
            pixel = original_image[i][j]
            new_pixel = (int(pixel[0] * brightness_multiplier),
                         int(pixel[1] * brightness_multiplier),
                         int(pixel[2] * brightness_multiplier))

            for pixel in new_pixel:
                if pixel > 255:
                    pixel = 255
                elif pixel < 0:
                    pixel = 0

            new_image_list.append(new_pixel)
    new_image.putdata(new_image_list)
    new_image.save('colour_brightness.png')
    insert_bin_file('dark_data', new_image)
    return new_image


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1]+list(crop_shape) +[3])
    return images


def read_from_bin_files(data_file, label_file, crop_size, idx):
    data = np.memmap(data_file, dtype='uint8', mode='r', shape=(81, 81, 3),
                     offset=crop_size[0] * crop_size[1] * crop_size[2] * idx)
    label = np.memmap(label_file, dtype='uint8', mode='r', shape=(1,), offset=idx)
    print(type(data))
    data_array = np.array(data)
    print(type(data_array))
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.imshow(data_array)
    return data, label


def main():
    train_data_set = load_tfl_data('../../my_big_data/train')
    print(len(train_data_set))
    for i, image in enumerate(train_data_set):
        flip(image)
        print(i)

    # path = '../../data_dir/cnn_min_course/train/'



    # for i, image in enumerate(train_data_set):
    #     image_array = np.array(image)
    #     mirror = ImageOps.mirror(image_array)
    #     fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    #     max_mag.imshow(mirror)
    #     print(i)


#-----dark images
    # for i, image in enumerate(train_data_set):
    #     change_brightness(image, 'darken', 80)
    #     print(i)
    plt.show(block=True)


if __name__ == '__main__':
    main()

# ax = \
# plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False, sharex=True,
#              sharey=True)[1]  # .flatten()
