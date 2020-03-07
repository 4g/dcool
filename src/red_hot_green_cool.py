import sys, random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

def add_ball(space, centerx, centery, vectorx, vectory, color):
    """Add a ball to the given space at a random position"""
    mass = 1
    radius = 3
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    x = centerx + random.randint(-radius, radius)
    body.position = x, centery + random.randint(-radius, radius)
    body.velocity = Vec2d(vectorx, vectory)
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.color = pygame.color.THECOLORS[color]
    space.add(body, shape)
    return shape


def add_L(space):
    """Add a inverted L shape with two joints"""
    body = space.static_body
    body.position = (300, 100)
    l1 = pymunk.Segment(body, (-150, 0), (655.0, 0.0), 5.0)
    l2 = pymunk.Segment(body, (-150, 200), (655.0, 200.0), 5.0)
    space.add(l1, l2)
    return l1,l2

def add_balls_at(space, balls, x, y, rate=5, vectorx=0, vectory=0, color='green'):
    for i in range(rate):
        ball_shape = add_ball(space, x, y, vectorx, vectory, color)
        balls.append(ball_shape)

def main():
    pygame.init()
    screen = pygame.display.set_mode((1000, 400))
    pygame.display.set_caption("Control the heat")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)

    lines = add_L(space)
    balls = []
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    heat_rate = 1
    ticks_to_next_ball = 10
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                heat_rate += 1

            elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                heat_rate -= 1

        ticks_to_next_ball -= 1
        if ticks_to_next_ball <= 0:
            ticks_to_next_ball = 5
            # add_balls_at(space, balls, 350, 400)
            add_balls_at(space, balls, 500, 175, rate=heat_rate, color='red')
            add_balls_at(space, balls, 500, 225, rate=heat_rate, color='red')

            add_balls_at(space, balls, 200, 200, rate=5, vectorx=50, vectory=0, color='cyan')
            add_balls_at(space, balls, 200, 150, rate=5, vectorx=100, vectory=0, color='cyan')
            add_balls_at(space, balls, 200, 250, rate=5, vectorx=50, vectory=0, color='cyan')

        screen.fill((255, 255, 255))

        balls_to_remove = []
        for ball in balls:
            if ball.body.position.x < 150 or ball.body.position.x > 955:
                balls_to_remove.append(ball)


        for ball in balls_to_remove:
            space.remove(ball, ball.body)
            balls.remove(ball)

        space.debug_draw(draw_options)

        space.step(1/50.0)

        pygame.display.flip()
        clock.tick(50)

if __name__ == '__main__':
    main()