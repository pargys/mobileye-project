import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from frame_container import *
from part1_light_spots_detection.run_attention import find_tfl_lights
from part3_estimated_distance_from_tfls.SFM import calc_TFL_dist
from part2_train_tfl_net.build_dataset import crop


class TFLMan:
    def __init__(self, principle_point, focal_length):
        self.__principal_point = principle_point
        self.__focal_length = focal_length
        self.__prev_frame = FrameContainer()
        self.__curr_frame = FrameContainer()
        self.__is_first_frame = True

    def on_frame(self, frame_index, frame_path, EM):
        self.__curr_frame.set_frame(index=frame_index, frame_path=frame_path, EM=EM)
        self.find_candidates()
        self.find_tfls()
        self.__curr_frame.check_tfl_validation()

        if not self.__is_first_frame:
            self.tfl_dist()
        self.visualize()
        self.__prev_frame = self.__curr_frame
        self.__is_first_frame = False

    def find_candidates(self):
        R, G = 0, 1
        red_x, red_y, green_x, green_y = find_tfl_lights(self.__curr_frame.image)
        check_coordinates_validation(red_x, red_y)
        check_coordinates_validation(green_x, green_y)
        self.__curr_frame.candidates = [[x, y] for x, y in zip(green_x, green_y)]
        self.__curr_frame.candidates += [[x, y] for x, y in zip(red_x, red_y)]
        self.__curr_frame.cand_auxiliary = [G for i in range(len(green_x))] + [R for i in range(len(red_x))]

    def find_tfls(self):
        #run the frame on the net
        candidates = self.__curr_frame.candidates
        image = self.__curr_frame.image
        auxiliary = self.__curr_frame.cand_auxiliary
        x, y = 0, 1
        candidates = np.array(candidates)
        crop_images = crop(image, candidates[:, x], candidates[:, y])
        loaded_model = load_model("../part2_train_tfl_net/Tzipi.h5")
        l_prediction = loaded_model.predict(crop_images)
        for index, pre in enumerate(l_prediction):
            if pre[1] > 0.8:
                self.__curr_frame.traffic_light.append(candidates[index])
                self.__curr_frame.traffic_light_auxiliary.append(auxiliary[index])

    def tfl_dist(self):
        calc_TFL_dist(self.__prev_frame, self.__curr_frame, self.__focal_length, self.__principal_point)

    def visualize(self):
        frame_path = self.__curr_frame.path
        index = self.__curr_frame.index
        candidates = self.__curr_frame.candidates
        cand_auxiliary = self.__curr_frame.cand_auxiliary
        traffic_lights = self.__curr_frame.traffic_light
        tfls_auxiliary = self.__curr_frame.traffic_light_auxiliary
        distances = self.__curr_frame.traffic_lights_3d_location[:, 2]

        # plot images
        tfl_img = open_mark_image(frame_path, traffic_lights, tfls_auxiliary)
        candidates_img = open_mark_image(frame_path, candidates, cand_auxiliary)
        fig, (candidate_plt, traffic_light_plt, dist_plt) = plt.subplots(3, 1, figsize=(12, 6))
        fig.suptitle(f'frame # {index} {frame_path}')
        candidate_plt.set_ylabel('candidates')
        candidate_plt.imshow(candidates_img)
        traffic_light_plt.set_ylabel('traffic_lights')
        traffic_light_plt.imshow(tfl_img)
        dist_plt.set_ylabel('distances')

        if not self.__is_first_frame:
            img = mark_dist(frame_path, traffic_lights, tfls_auxiliary, distances)
            dist_plt.set_ylabel('dist')
            dist_plt.imshow(img)

        plt.show(block=True)


def check_coordinates_validation(coord_x, coord_y):
    try:
        assert len(coord_x) == len(coord_y)
    except AssertionError:
        coord_x.clear()
        coord_y.clear()


def open_mark_image(frame_path, lights_coords, auxiliary):
    image = np.array(Image.open(frame_path))
    return mark_pixels(image, lights_coords, auxiliary)


def mark_pixels(image, lights_coords, auxiliary):
    x, y = 0, 1
    G = 1
    marker_size = 8
    for coord, color in zip(lights_coords, auxiliary):
        color = [0, 255, 0] if color == G else [255, 0, 0]
        image[coord[x]-marker_size: coord[x]+marker_size+1, coord[y]-marker_size: coord[y]+marker_size+1] = color
    return image


def mark_dist(frame_path, traffic_lights, auxiliary, distance):
    image = Image.open(frame_path)
    d = ImageDraw.Draw(image)
    fnt = ImageFont.truetype('..font/ariblk.ttf', 40)

    for row, dist in zip(traffic_lights, distance):
        d.text((row[1], row[0]), r'{0:.1f}'.format(dist), font=fnt, fill=(255, 255, 0))
    image = np.array(image)
    image = mark_pixels(image, traffic_lights, auxiliary)
    return image
