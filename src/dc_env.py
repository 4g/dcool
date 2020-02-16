import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d


class BallProducer:
    def __init__(self, centerx, centery, vel_x, vel_y, rate, color, env):
        self.centerx = centerx
        self.centery = centery
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.rate = rate
        self.color = color
        self.env = env

    @staticmethod
    def get_ball(centerx, centery, vel_x, vel_y, color, approx=True, mass=1, radius=5):
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
        return body, shape

    def step(self):
        for i in range(self.rate):
            ball = BallProducer.get_ball(centerx=self.centerx,
                                 centery=self.centery,
                                 color=self.color,
                                 vel_x=self.vel_x,
                                 vel_y=self.vel_y)
            self.env.add_ball(ball)

class DataCenterEnv:
    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.walls = []
        self.heaters = []
        self.fans = []
        self.balls = []

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

    def add_ball(self, ball):
        self.space.add(ball)
        self.balls.append(ball[1])

    def add_wall(self, x, y, l, thickness=5.0):
        body = self.space.static_body
        body.position = (x, y)
        l1 = pymunk.Segment(body, (0, 0), (l, 0.0), thickness)
        self.space.add(l1)
        self.walls.append(l1)

    def add_heater(self, x, y, rate=5):
        heater = BallProducer(x, y, 0, 0, rate=rate, color='red', env=self)
        self.heaters.append(heater)
        return heater

    def add_fan(self, x, y, rate=5):
        fan = BallProducer(x, y, 30, 0, rate=rate, color='green', env=self)
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

    def run(self):
        # run the data center
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Control the heat")
        clock = pygame.time.Clock()

        draw_options = pymunk.pygame_util.DrawOptions(screen)
        iter = 0
        while True:
            iter += 1
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

            screen.fill((255, 255, 255))
            self.cleanup()
            self.space.debug_draw(draw_options)
            self.space.step(1 / 60.0)

            pygame.display.flip()
            clock.tick(60)

    def action(self):
        pass

    def state(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass


dc = DataCenterEnv(800, 500)

dc.add_wall(100, 150, 600)
dc.add_wall(100, 350, 600)

dc.add_fan(150, 175, rate=3)
dc.add_fan(150, 325, rate=3)

dc.add_heater(450, 250, rate=5)

dc.run()