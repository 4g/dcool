import numpy as np
import cv2

size = (500, 500)
frame = np.zeros(size)
frame = frame + 10

num_servers = 5
server_temperatures = [150, 200, 480, 220, 900]
server_location = [(30, 30), (80, 30), (160, 100), (200, 200), (270, 30)]
server_fan_speed = lambda temp: temp/10
server_output = lambda inp, ins: (inp + ins)/2


def server_fanout(fan_speed):
    x = np.arange(1, size[0], 1)
    y = np.arange(1, size[1], 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (fan_speed)  / (xx ** 2.0 + yy ** 2.0)
    return z

def server_frame(size, server_location, server_temperature):
    s_frame = np.zeros(size)
    server_input_temp = frame[server_location[0], server_location[1] - 5]
    server_output_temp = server_output(server_input_temp, server_temperature)   
    fs = server_fan_speed(server_temperature)
    s_fanout = server_fanout(fs) * server_output_temp
    s_fanout[s_fanout > 128] = 128
    s_frame[server_location[0]:server_location[0]+s_fanout.shape[0], server_location[1]:server_location[1]+s_fanout.shape[1]] = s_fanout[0:-server_location[0]+1, 0:-server_location[1]+1]
    server_temperature = (server_temperature + server_input_temp*0.005*(fs**0.05))/(0.005*fs**0.05 + 1)
    return s_frame, server_temperature

for t in range(10000):
    temp_frame = None
    for index in range(num_servers):
        s_frame, temperature = server_frame(size, server_location[index], server_temperatures[index])
        server_temperatures[index] = temperature
        if temp_frame is None:
            temp_frame = s_frame

        temp_frame = (temp_frame + s_frame/num_servers)

    frame[temp_frame != 0] = (temp_frame[temp_frame != 0] + frame[temp_frame != 0]) / 2
    frame = cv2.blur(frame, (2, 2))

    display = cv2.applyColorMap(np.asarray(frame, dtype=np.uint8), cv2.COLORMAP_JET)
    # print (server_temperatures)

    cv2.imshow("win", display)
    cv2.waitKey(1)
