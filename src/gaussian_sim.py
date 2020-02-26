import scipy
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from tqdm import tqdm



class Field:
    def get_state(self, time):
        



x, y = np.meshgrid(np.linspace(-2, 1, 100), np.linspace(-4, 1, 100))
d = np.sqrt(x*x+y*y)

x1, y1 = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
d1 = np.sqrt(x1*x1+y1*y1)

def get_gaussian(d, sigma, mu, channel):
    g = np.exp(-((d-mu)**2/(2.0*sigma**2)))
    ge = np.expand_dims(g, 0)
    gc = np.concatenate([ge, ge, ge])
    gc[channel] = gc[channel] * 2
    return gc


for i in tqdm(range(1000)):
    gs1 = get_gaussian(d, i/500.0, 0.0, 0)
    gs2 = get_gaussian(d1, i/500.0, 0.0, 2)
    gs = (gs1 + gs2) / 2
    gs = np.swapaxes(gs, 0, -1)
    gs = np.asarray(gs * 128, dtype=np.uint8)
    gs = cv2.resize(gs, (500, 500))
    cv2.imshow("wind", gs)
    cv2.waitKey(1)


class GaussianDCEnv:
    def __init__(self):
        pass

    def add_heater(self, x, y, rate):
        pass

    def add_fan(self, x, y, rate):
        pass

    def actions(self):
        pass

    def update_field(self):
        fields = []
        for element in self.elements:
            field = element.field()
            fields.append(field)

        # current_field =

    def step(self):
        self.update_field()
        self.update_sensors()
        self.update_metrics()
        return self.current_field