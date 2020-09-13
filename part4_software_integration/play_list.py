import pickle


class PlayList:
    def __init__(self, list_path):
        with open(list_path + '.pls') as play_list_file:
            play_list = play_list_file.read().split('\n')

        pkl_path = play_list[0]
        with open(pkl_path, 'rb') as pkl_file:
            self.data = pickle.load(pkl_file, encoding='latin1')
        self.first_frame_id = int(play_list[1])
        self.frame_list = play_list[2:]
