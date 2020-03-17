import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import threading
import numpy as np
from matplotlib import cm
import gym
from gym import spaces
from collections import OrderedDict

colormap = cm.get_cmap('jet', 256)

collision_types = {
    "ball": 0,
    "rack": 1,
    "ahu": 2,
    "wall": 3,
    "tile": 4,
}

def temp_to_color(x):
    return tuple([int(i*255) for i in colormap(x)])

class Rack:
    def __init__(self, env, x, y, l, name, thickness=3.0, direction=1, temperature=.75):
        self.env = env
        self.direction = direction
        self.thickness = thickness
        self.direction = direction
        self.temperature = temperature
        self.x = x
        self.y = y
        self.l = l
        self.name = name
        self.body = self.add_body()

    def add_body(self):
        body = self.env.space.static_body
        body.position = (self.x, self.y)
        l1 = pymunk.Segment(body, (0, 0), (self.l, 0.0), self.thickness)
        l1.color = temp_to_color(self.temperature)
        l1.collision_type = collision_types["rack"]
        l1.name = self.name
        l1.temperature = self.temperature
        self.env.space.add(l1)
        return l1

    def set_temperature(self, temperature):
        self.temperature = temperature
        self.body.temperature = temperature
        self.body.color = temp_to_color(temperature)

    def step(self):
        x = self.x + self.l / 2
        y = self.y + (-4 * self.thickness if self.direction == -1 else 2 * self.thickness)
        rate = 3 if self.temperature > 0.5 else 2 if self.temperature > 0.2 else 1
        for i in range(rate):
            vel_x = 0
            vel_y = 10 * self.direction
            ball = BallProducer.get_ball(centerx=x,
                                         centery=y,
                                         temperature=self.temperature,
                                         vel_x=vel_x,
                                         vel_y=vel_y)

            self.env.add_ball(ball)

class Tile:
    def __init__(self, env, x, y, l, thickness=3.0, rate=2, temperature=.75):
        self.env = env
        self.thickness = thickness
        self.temperature = temperature
        self.rate = rate
        self.x = x
        self.y = y
        self.l = l
        self.add_body()

    def add_body(self):
        body = self.env.space.static_body
        body.position = (self.x, self.y)
        l1 = pymunk.Segment(body, (0, 0), (self.l, 0.0), self.thickness)
        l1.color = pygame.color.THECOLORS['pink']
        l1.collision_type = collision_types["tile"]
        self.env.space.add(l1)

    def step(self):
        for i in range(self.rate):
            vel_x = random.randint(-10, 10)
            vel_y = random.randint(-3, 3)
            ball = BallProducer.get_ball(centerx=self.x,
                                         centery=self.y,
                                         temperature=self.temperature,
                                         vel_x=vel_x,
                                         vel_y=vel_y)

            self.env.add_ball(ball)
            self.env.tile_energy += 1/self.temperature

class BallProducer:
    @staticmethod
    def get_ball(centerx, centery, vel_x, vel_y, temperature, approx=True, mass=1, radius=5):
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
        shape.temperature = temperature
        shape.color = temp_to_color(temperature)
        shape.collision_type = collision_types["ball"]
        shape.name = "ball"
        return body, shape

class DataCenter(threading.Thread):
    def __init__(self, width, height, render=True):
        super(DataCenter, self).__init__()
        threading.Thread.__init__(self)
        self.width = width
        self.height = height
        self.minx = width
        self.miny = height
        self.maxx = 0
        self.maxy = 0
        self.render = render

        self.walls = []
        self.racks = []
        self.tiles = []
        self.objects = {}

        self.balls = []

        self.collided = []
        self.tile_energy = 0
        self.ahu_energy = 0
        self.total_energy = 0
        self.max_rack_temp = 0
        self.running = False

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.generations = 1

        if self.render:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            self.screen = screen
            self.font = pygame.font.SysFont("Arial", 24)

        collisions_allowed = [("ball", "rack"), ("ball", "ball")]
        for x1, x2 in collisions_allowed:
            h = self.space.add_collision_handler(
                collision_types[x1],
                collision_types[x2])

            h.separate = self.collision_handler

    def collision_handler(self, arbiter, space, data):
        shape1 = arbiter.shapes[0]
        shape2 = arbiter.shapes[1]
        self.collided.append((shape1, shape2))

    def add_ball(self, ball):
        self.space.add(ball)
        self.balls.append(ball[1])

    def add_wall(self, x, y, l, vertical=False):
        self.minx = min(x, self.minx)
        self.miny = min(y, self.miny)

        if vertical:
            self.maxy = max(y+l, self.miny)
        else:
            self.maxx = max(x + l, self.maxx)

    def add_rack(self, x, y, l, thickness, direction):
        name = "rack" + str(len(self.racks))
        rack = Rack(self, x, y, l, temperature=.9, thickness=thickness, direction=direction, name=name)
        self.objects[name] = rack
        self.racks.append(rack)

    def add_tile(self, x, y, l, thickness=3.0):
        tile = Tile(self, x, y, l, thickness, rate=0, temperature=0.1)
        self.tiles.append(tile)

    def post_step_update(self):
        removed = set()
        for x1, x2 in self.collided:
            c2 = x2.collision_type
            if c2 == collision_types["rack"]:
                ball_temperature = x1.temperature
                rack_temperature = self.objects[x2.name].temperature
                if rack_temperature > ball_temperature:
                    rack_temp = (self.objects[x2.name].temperature + 0.1*ball_temperature)/1.1
                    new_rack_temp = rack_temp
                    self.objects[x2.name].set_temperature(new_rack_temp)

                x1id = x1._get_shapeid()
                if x1id not in removed:
                    self.space.remove(x1, x1.body)
                    removed.add(x1id)

            elif c2 == collision_types["ball"]:
                nt1 = (x1.temperature + 0.5 * x2.temperature) / 1.5
                nt2 = (x2.temperature + 0.5 * x1.temperature) / 1.5
                x1.temperature = nt1
                x1.color = temp_to_color(nt1)
                x2.temperature = nt2
                x2.color = temp_to_color(nt2)


        self.collided = []

        balls_to_remove = []
        for ball in self.balls:
            if ball.body.position.x <= self.minx or ball.body.position.x >= self.maxx:
                balls_to_remove.append(ball)
            if ball.body.position.y <= self.miny or ball.body.position.y >= self.maxy:
                balls_to_remove.append(ball)

        self.ahu_energy = sum([1/ball.temperature for ball in balls_to_remove])

        self.total_energy += self.ahu_energy + self.tile_energy
        self.tile_energy = 0

        for ball in balls_to_remove:
            self.space.remove(ball, ball.body)
            self.balls.remove(ball)

        self.max_rack_temp = max([r.temperature for r in self.racks])

    def step(self):
        for tile in self.tiles:
            tile.step()

        for rack in self.racks:
            rack.step()

    def stop(self):
        self.running = False

    def run(self):
        # run the data center
        self.running = True

        clock = pygame.time.Clock()
        if self.render:
            pygame.display.set_caption("Control the heat")
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        iter = 0
        while self.running:
            self.generations += 1
            if self.render:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        sys.exit(0)

            self.step()
            self.post_step_update()
            if self.render:
                self.screen.fill((0, 0, 0))
                self.space.debug_draw(draw_options)

            self.space.step(1)
            if self.render:
                self.screen.blit(self.font.render("Steps:" + str(self.generations) +
                                                  " Max temp:"+ str(round(self.max_rack_temp,1)) +
                                                  " Energy:" + str(int(self.total_energy)),
                                                  1, pygame.color.THECOLORS["white"]), (0, 0))

                pygame.display.flip()
            clock.tick(60)

def get_dc(render):
    dc = DataCenter(800, 600, render=render)
    dc.add_wall(70, 50, 635)
    dc.add_wall(70, 550, 635)
    dc.add_wall(70, 50, 500, vertical=True)
    dc.add_wall(705, 50, 500, vertical=True)


    rack_xs = [100, 200, 300, 400, 500, 600]
    rack_ys = [150, 250, 350, 450]

    for x in rack_xs:
        for index, y in enumerate(rack_ys):
            dc.add_rack(x, y, 75, thickness=6, direction= -1 + (index%2)*2)
            if index%2 == 0:
                dc.add_tile(x, y+50, 10)

    return dc

import  time

class DataCenterEnv(gym.Env):
    metadata = {}

    def __init__(self, render=False):
        super(DataCenterEnv, self).__init__()
        self.render = render
        self.DC = None
        self.setup()

    def step(self, action):
        # for t, r in zip(action, self.DC.tiles):
        #     r.temperature = t/10.0

        for i, r in enumerate(self.DC.tiles):
            if i == action:
                self.DC.tiles[i].temperature = 0.1
                self.DC.tiles[i].rate = 5
            else:
                self.DC.tiles[i].temperature = 0.5
                self.DC.tiles[i].rate = 0
        time.sleep(.01)
        observation = self.get_observation()
        reward = 0

        done = 0
        # print (self.DC.max_rack_temp, observation)
        if self.DC.max_rack_temp <= 0.8:
            done = 1
            reward = (10000 - self.DC.total_energy) / self.DC.generations
        info = {}

        return observation, reward, done, info

    def get_observation(self):
        r_t = [r.temperature for r in self.DC.racks]
        return r_t

    def setup(self):
        self.DC = get_dc(self.render)
        num_tiles = len(self.DC.tiles)
        num_racks = len(self.DC.racks)
        self.action_space = spaces.Discrete(num_tiles)
        self.observation_space = spaces.MultiDiscrete([10 for i in range(num_racks)])


    def stop(self):
        if self.DC is not None and self.DC.running:
            self.DC.stop()
            self.DC.join()

    def reset(self):
        self.stop()
        self.setup()
        self.DC.start()

        return self.observation_space.sample()

    def close (self):
        pass

