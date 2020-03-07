import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import threading
from gym.spaces import MultiDiscrete
import json
import subprocess
import gym
from gym import spaces
import numpy as np


class Metrics:
    def __init__(self):
        self.data = {}

    def add(self, name, value):
        self.data[name] = self.data.get(name, [])
        self.data[name].append(value)

    def update(self, d):
        for k in d:
            self.add(k, d[k])

    def latest(self):
        l = {}
        for name in self.data:
            l[name] = self.data[name][-1]
        return l

    def __str__(self):
        return json.dumps(self.data)

    def __repr__(self):
        return str(self.data)

class BallProducer:
    def __init__(self, centerx, centery, vel_x, vel_y, rate, color, env, name):
        self.centerx = centerx
        self.centery = centery
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.rate = rate
        self.color = color
        self.env = env
        self.name = name


    @staticmethod
    def get_ball(centerx, centery, vel_x, vel_y, color, approx=True, mass=1, radius=3, name=None):
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = centerx
        y = centery
        if approx:
            x += random.randint(-radius, radius)
            y += random.randint(-radius, radius)
        body.position = x, y
        body.velocity = Vec2d(vel_x, vel_y)
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.color = pygame.color.THECOLORS[color]
        shape.name = name
        return body, shape

    def step(self):
        for i in range(self.rate):
            ball = BallProducer.get_ball(centerx=self.centerx,
                                 centery=self.centery,
                                 color=self.color,
                                 vel_x=self.vel_x,
                                 vel_y=self.vel_y,
                                 name=self.name)

            self.env.add_ball(ball)


class DataCenterEnv(threading.Thread, gym.Env):
    def __init__(self, width, height, render=True):
        super(DataCenterEnv, self).__init__()
        threading.Thread.__init__(self)
        self.width = width
        self.height = height
        self.render = render

        self.walls = []
        self.heaters = []
        self.fans = []
        self.balls = []

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.speed_counter = {}
        self.energy_counter = {}
        self.metrics = Metrics()
        self.action_space = MultiDiscrete([10 for i in self.fans])
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(width, height, 3), dtype=np.uint8)

        if self.render:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            self.screen = screen

    def add_ball(self, ball):
        self.space.add(ball)
        self.balls.append(ball[1])
        self.increment_energy(ball[1])

    def increment_energy(self, ball):
        etype = ball.name + "_energy"
        self.energy_counter[etype] = self.energy_counter.get(etype, 0) + 1

    def add_wall(self, x, y, l, thickness=5.0, vertical=False):
        body = self.space.static_body
        body.position = (x, y)
        if vertical:
            l1 = pymunk.Segment(body, (0, 0), (0.0, l), thickness)
        else:
            l1 = pymunk.Segment(body, (0, 0), (l, 0.0), thickness)
        self.space.add(l1)
        self.walls.append(l1)

    def add_heater(self, x, y, rate=5):
        heater = BallProducer(x, y, 0, 0, rate=rate, color='red', env=self, name='heater')
        self.heaters.append(heater)
        return heater

    def add_fan(self, x, y, rate=5):
        fan = BallProducer(x, y, 100, 0, rate=rate, color='green', env=self, name='fan')
        self.fans.append(fan)
        return fan

    def cleanup(self):
        balls_to_remove = []
        for ball in self.balls:
            if ball.body.position.x < 100 or ball.body.position.x > 700:
                balls_to_remove.append(ball)

        for ball in balls_to_remove:
            self.space.remove(ball, ball.body)
            self.balls.remove(ball)

        self.speed_counter = {}
        for ball in self.balls:
            key = ball.name + "_presence"
            self.speed_counter[key] = self.speed_counter.get(key, 0) + 1

    def run(self):
        # run the data center

        clock = pygame.time.Clock()
        if self.render:
            pygame.display.set_caption("Control the heat")
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        iter = 0
        while True:
            iter += 1
            if self.render:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        sys.exit(0)

            if iter % 5 == 0:
                for heater in self.heaters:
                    heater.step()

                for fan in self.fans:
                    fan.step()

            if self.render:
                self.screen.fill((255, 255, 255))
                for fan in self.fans:
                    pygame.draw.rect(self.screen, pygame.color.THECOLORS['blue'],
                                     pygame.Rect(fan.centerx - 15, self.height - fan.centery - 15, 30, 30))
                for heater in self.heaters:
                    pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'],
                                     pygame.Rect(heater.centerx - 15, self.height - heater.centery - 15, 30, 30))



            self.cleanup()
            if self.render:
                self.space.debug_draw(draw_options)

            self.space.step(1 / 60.0)
            if self.render:
                pygame.display.flip()
            clock.tick(60)

    def step(self, action):
        for fan, rate in zip(self.fans, action):
            fan.rate = rate

        return pygame.surfarray.pixels2d(self.screen), self.metrics, False, None


    def reset(self):
        return None

    def update_metrics(self):
        self.metrics.update(self.energy_counter)
        self.metrics.update(self.speed_counter)

        for i, fan in enumerate(self.fans):
            self.metrics.add("fan_speed_" + str(i), fan.rate)

        for i, heater in enumerate(self.heaters):
            self.metrics.add("heater_speed_" + str(i), heater.rate)

    def state(self):
        self.update_metrics()
        return self.metrics.latest()