import scipy
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from tqdm import tqdm

class Field:
    def __init__(self, shape, centerx, centery, rate):
        self.shape = shape
        self.rate = rate
        self.centerx = centerx
        self.centery = centery
        x, y = np.meshgrid(np.linspace(-1 + 2*centerx, 1, self.shape[0]), np.linspace(-1 + 2*centery, 1, self.shape[1]))
        self.d = np.sqrt(x * x + y * y)
        self.state = np.zeros(self.shape)
        self.time = 0
        self.step()

    def step(self):
        self.state = self.get_gaussian(self.d, self.rate * self.time / 5000.0, 0.0)
        self.time += 1
        return self.state

    @staticmethod
    def get_gaussian(d, sigma, mu):
        g = np.exp(-((d-mu)**2/(2.0*sigma**2)))
        ge = np.expand_dims(g, 0)
        return ge

    # for i in tqdm(range(1000)):
    #     gs = np.swapaxes(gs, 0, -1)
    #     gs = np.asarray(gs * 128, dtype=np.uint8)
    #     gs = cv2.resize(gs, (500, 500))
    #     cv2.imshow("wind", gs)
    #     cv2.waitKey(1)


class GaussianDCEnv:
    def __init__(self):
        self.field_shape = (500, 500)
        self.space_shape = (500, 500)
        self.heaters = []
        self.fans = []
        self.heater_channel = 1
        self.fan_channel = 2
        self.current_field = None

    def add_heater(self, x, y, rate):
        heat_field = Field(self.field_shape, x, y, rate)
        self.heaters.append(heat_field)

    def add_fan(self, x, y, rate):
        heat_field = Field(self.field_shape, x, y, rate)
        self.fans.append(heat_field)

    def add_sensor(self, x, y):
        pass

    def actions(self):
        pass

    def update_sensors(self):
        pass

    def update_metrics(self):
        pass

    def update_field(self):
        fields = np.zeros((self.field_shape[0], self.field_shape[1], 3))
        heat_fields = np.asarray([heater.step() for heater in self.heaters])
        heat_field = np.average(heat_fields, axis=0)
        fields[:, :, self.heater_channel] = heat_field

        fan_fields = np.asarray([fan.step() for fan in self.fans])
        fan_field = np.average(fan_fields, axis=0)
        fields[:, :, self.fan_channel] = fan_field

        fields[:, :, self.heater_channel] =

        if self.current_field is None:
            self.current_field = fields
        else:
            self.current_field = (self.current_field + fields)/2.0

        tmp = np.asarray(255*self.current_field, dtype=np.uint8)
        cv2.imshow("win", tmp)
        cv2.waitKey(1)

    def step(self):
        self.update_field()
        self.update_sensors()
        self.update_metrics()
        return self.current_field

gsdc = GaussianDCEnv()
for i in range(4):
    gsdc.add_heater(i*0.2, 0, 1)


for i in tqdm(range(100)):
    gsdc.update_field()

for i in range(4):
    gsdc.add_fan(i*0.2, 0.2, 1)

for i in tqdm(range(100)):
    gsdc.update_field()
