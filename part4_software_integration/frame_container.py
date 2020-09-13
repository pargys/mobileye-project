import numpy as np
from PIL import Image


class FrameContainer:
    def __init__(self):
        self.image = []
        self.index = None
        self.path = None
        self.candidates = []
        self.cand_auxiliary = []
        self.traffic_light = []
        self.traffic_light_auxiliary = []
        self.traffic_lights_3d_location = np.array([np.zeros(3)])
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]

    def check_tfl_validation(self):
        try:
            assert (len(self.candidates) >= len(self.traffic_light))
        except AssertionError:
            print(f"Something went wrong with traffic light detection.")
            self.traffic_light = self.candidates

    def set_frame(self, index, frame_path, EM):
        self.path = frame_path
        self.image = np.array(Image.open(frame_path))
        self.index = index
        self.EM = EM

