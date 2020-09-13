import numpy as np

from tfl_manager import TFLMan
from play_list import PlayList


class Controller:
    def __init__(self, play_lists_file):
        with open(play_lists_file + '.pls') as play_list_file:
            play_lists_path = play_list_file.read().split('\n')
        self.__play_lists = []

        for list_path in play_lists_path:
            self.__play_lists.append(PlayList(list_path))

    def run(self):
        for pls in self.__play_lists:
            tfl_man = TFLMan(pls.data['principle_point'], pls.data['flx'])
            for index, img_path in enumerate(pls.frame_list, pls.first_frame_id):
                EM = calc_EM(pls.data, index) if index > pls.first_frame_id else None
                tfl_man.on_frame(index, img_path, EM)


def calc_EM(data,  frame_index):
    EM = np.eye(4)
    for i in range(frame_index-1, frame_index):
        EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
    return EM


def main():
    controller = Controller('data/pls_files/‏‏‫play_lists')
    controller.run()


if __name__ == '__main__':
    main()





