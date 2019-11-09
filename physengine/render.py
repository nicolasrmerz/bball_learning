import pymunk
import pyglet
from pymunk.pyglet_util import DrawOptions
import time
from math import degrees, cos, sin
import random
import threading
import queue
import os

q = queue.Queue()
dirname = os.path.dirname(os.path.abspath(__file__))

class Action():
    def __init__(self, action_type, velocity=0, angle=0):
        self.action_type = action_type
        self.velocity = velocity
        self.angle = angle
        
class PymunkSpace():
    def __init__(self, winwidth, winheight):
        self.winwidth = winwidth
        self.winheight = winheight
        # Define some constant constraints for the system
        self.MIN_NET_X = self.winwidth / 2 + 30
        self.MAX_NET_X = self.winwidth - 50
        self.MIN_NET_Y = 200
        self.MAX_NET_Y = self.winheight - 100
        
        self.MIN_BALL_X = 50
        self.MAX_BALL_X = self.winwidth / 2 - 30
        self.STARTING_BALL_Y = 200
        
        self.WIND_FORCE = 200
        
        self.DEFAULT_GRAVITY = -900
        
        self.reset_space(True)
        
    def create_circle(self, mass, radius, posx, posy, label, body_type):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type)
        #Initially start with basketball being kinetic, as we want to be able to freely control it before taking the shot
        #self.basketball_body = pymunk.Body(body_type=KINETIC)
        shape = pymunk.Circle(body, radius)
        #self.basketball_body.position = 300, 325
        body.position = posx, posy
        # Just use these values for all objects; adjust if needed
        shape.elasticity = 0.95
        shape.friction = 0.5
        # Used to see if there's a collision between the basketball and the ground (shot was missed)
        shape.id = label
        return body, shape
        
    def create_segment(self, p1x, p1y, p2x, p2y, label, thickness):
        body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (p1x, p1y), (p2x, p2y), thickness)
        shape.id = "connector"
        shape.friction = 0.5
        shape.elasticity = 0.95
        return body, shape
        
    def reset_space(self, doRandomize):
        self.space = pymunk.Space()
        self.handler = self.space.add_default_collision_handler();
        self.handler.begin = self.coll_begin
        self.handler.pre_solve = self.coll_pre
        self.handler.post_solve = self.coll_post
        self.handler.separate = self.coll_separate
        self.ball_missed = False
        self.ball_hit = False
        
        if doRandomize:
            self.initial_basketball_x = random.randrange(self.MIN_BALL_X, self.MAX_BALL_X, 10)
            # Basketball will always start at the same height
            self.initial_basketball_y = self.STARTING_BALL_Y
            
            self.initial_hoop_x = random.randrange(self.MIN_NET_X, self.MAX_NET_X, 10)
            self.initial_hoop_y = random.randrange(self.MIN_NET_Y, self.MAX_NET_Y, 10)
            
            wind_angle = random.randrange(0, 360, 15)
            self.wind_x, self.wind_y = self.get_x_y(self.WIND_FORCE, wind_angle)


        self.space.gravity = 0 + self.wind_x, self.DEFAULT_GRAVITY + self.wind_y

        self.mass = 1
        self.ball_radius = 15

        self.basketball_body, self.basketball_shape = self.create_circle(self.mass, self.ball_radius, 
            self.initial_basketball_x, self.initial_basketball_y, "basketball", pymunk.Body.STATIC)
        
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_shape = pymunk.Poly.create_box(self.ground_body, size=(10000, 100))
        # Used to see if there's a collision between the basketball and the ground (shot was missed)
        self.ground_shape.id = "ground"
        self.ground_body.position = 0, 0
        self.ground_shape.elasticity = 0.95
        self.ground_shape.friction = 0.5
        
        self.generate_net(self.ball_radius, self.initial_hoop_x, self.initial_hoop_y)
        
        self.space.add(self.basketball_body, self.basketball_shape)
        self.space.add(self.ground_body, self.ground_shape)
        self.space.add(self.rim1_body, self.rim1_shape)
        self.space.add(self.rim2_body, self.rim2_shape)
        self.space.add(self.connector_body, self.connector_shape)
        self.space.add(self.backboard_body, self.backboard_shape)
            
    def coll_begin(self, arbiter, space, data):
        if arbiter.shapes[0].id == "basketball" and arbiter.shapes[1].id == "ground":
            self.ball_missed = True
        return True
        
    def coll_pre(self, arbiter, space, data):
        return True
        
    def coll_post(self, arbiter, space, data):
        pass
        
    def coll_separate(self, arbiter, space, data):
        pass
    
    def get_x_y(self, velocity, angle):
        vel_x = velocity * cos(angle)
        vel_y = velocity * sin(angle)
        return vel_x, vel_y
    
    def generate_net(self, ball_radius, netx, nety):
        # Regulation basketball has a radius of 4.7 inches, and the rim has a radius of 9 inches.
        # This is a ratio of approximately 1.915
        rim_radius = ball_radius * 1.915
        # The metal has a diameter of 0.35 inches, which has a ratio of 0.075 when compared to the radius of the basketball
        metal_radius = ball_radius * 0.075
        # The connector is approximately 6 inches, which has a ratio of 1.277 when compared to the radius of the basketball
        connector_length = ball_radius * 1.277
        # Thickness not important, just set to a 10th of the radius of the ball
        connector_thickness = ball_radius * 0.1
        # Backboard is about 36 inches from the rim to the top, which has a ratio of 7.66 when compared to the radius of the basketball
        backboard_height = ball_radius * 7.66
        backboard_thickness = ball_radius * 0.1
        
        self.netx = netx
        self.nety = nety

        self.rim1_body, self.rim1_shape = self.create_circle(0, metal_radius, netx - rim_radius, nety, "rim1", pymunk.Body.STATIC)
        
        self.rim2_body, self.rim2_shape = self.create_circle(0, metal_radius, netx + rim_radius, nety, "rim2", pymunk.Body.STATIC)


        # There's a little straight piece of metal that connects the start of the rim to the backboard       
        self.connector_body, self.connector_shape = self.create_segment(netx + rim_radius, nety, netx + rim_radius + connector_length, nety, "connector", connector_thickness)
        
        self.backboard_body, self.backboard_shape = self.create_segment(netx + rim_radius + connector_length, nety, netx + rim_radius + connector_length, nety + backboard_height, "backboard", backboard_thickness)

        
    def check_bounds(self):
        if(self.basketball_body.position.x > self.rim1_body.position.x and self.basketball_body.position.x < self.rim2_body.position.x 
            and self.basketball_body.position.y < self.rim1_body.position.y + 10 and self.basketball_body.position.y > self.rim1_body.position.y - 10) and self.basketball_body.velocity.y < 0:
            self.ball_hit = True
            
    def shoot(self, velocity, angle):
        initial_vel_x, initial_vel_y = self.get_x_y(velocity, angle)
        
        posx = self.basketball_body.position.x
        posy = self.basketball_body.position.y

        self.space.remove(self.basketball_body)
        self.space.remove(self.basketball_shape)
        
        self.basketball_body, self.basketball_shape = self.create_circle(self.mass, self.ball_radius, 
            posx, posy, "basketball", pymunk.Body.DYNAMIC)

        self.basketball_body.velocity = initial_vel_x, initial_vel_y
        self.space.add(self.basketball_body)
        self.space.add(self.basketball_shape)
        
    def move(self, dir):
        posx = self.basketball_body.position.x
        posy = self.basketball_body.position.y
        # Don't want to be able to shoot past the halfway point of the screen
        if (posx >= self.winwidth/2 and dir == "r") or (posx <= 0 and dir == "l"):
            return

        self.space.remove(self.basketball_body)
        self.space.remove(self.basketball_shape)
        
        if dir == "r":
            self.basketball_body.position = posx + 10, posy
        else:
            self.basketball_body.position = posx - 10, posy

        self.space.add(self.basketball_body)
        self.space.add(self.basketball_shape)    
            
    
        
class PhysWin(pyglet.window.Window):
    def __init__ (self, winwidth, winheight):
        self.winheight = winheight
        self.winwidth = winwidth
        pyglet.window.Window.__init__(self, self.winheight, self.winwidth, fullscreen = False)
        self.options = DrawOptions()
        self.pmSpace = PymunkSpace(self.winwidth, self.winheight)
        background = pyglet.image.SolidColorImagePattern((255,255,255,255)).create_image(self.winheight, self.winwidth)
        self.background_sprite = pyglet.sprite.Sprite(background, x=0, y=0)
        pyglet.clock.schedule_interval(self.update, 1.0/60)
        self.doingAction = False
        #pyglet.clock.schedule_interval(self.check_queue, 5)

        hoop_img = pyglet.image.load(dirname + "/../resource/hoop_resized.png")
        hoop_img.anchor_x = hoop_img.width // 2 - 1
        hoop_img.anchor_y = hoop_img.height // 2 - 1
        self.hoop_sprite = pyglet.sprite.Sprite(hoop_img, x=0, y=0)
        
        self.reset_space(True)
        
        basketball_img = pyglet.image.load(dirname + "/../resource/basketball_resized.png")
        basketball_img.anchor_x = basketball_img.width // 2 - 1
        basketball_img.anchor_y = basketball_img.height // 2 - 1
        self.basketball_sprite = pyglet.sprite.Sprite(basketball_img, x=self.pmSpace.basketball_body.position.x, y=self.pmSpace.basketball_body.position.y)
           
    def reset_space(self, doReset):
        self.pmSpace.reset_space(doReset)
        self.hoop_sprite.update(self.pmSpace.netx-1, self.pmSpace.nety - 1 - 1.5*self.pmSpace.ball_radius)
        
        
    def check_bounds(self):
        self.pmSpace.check_bounds()

            
    def on_draw(self):
        self.clear()
        self.background_sprite.draw()
        self.pmSpace.space.debug_draw(self.options)
        self.basketball_sprite.draw()
        self.hoop_sprite.draw()
        
    def update(self, dt):
        if not q.empty() and not self.doingAction:
            item = q.get(block=False)
            self.doingAction = True
            if item.action_type == "shoot":
                self.shoot(item)
            elif item.action_type == "r":
                self.move("r")
            elif item.action_type == "l":
                self.move("l")
        self.pmSpace.space.step(dt)
        self.check_bounds()
        self.basketball_sprite.update(self.pmSpace.basketball_body.position.x, self.pmSpace.basketball_body.position.y, degrees(-self.pmSpace.basketball_body.angle))
        if self.pmSpace.ball_hit:
            print("Basket made!")
            self.reset_space(True)
            doingAction = False
        if self.pmSpace.ball_missed:
            print("Basket missed!")
            # Reset the space but don't move the starting position of the ball or the hoop
            self.reset_space(False)
            doingAction = False
            
    def shoot(self, item):
        self.pmSpace.shoot(item.velocity, item.angle)
            
    def move(self, dir):
        self.pmSpace.move(dir)
        self.doingAction = False

def test():
    time.sleep(2)
    q.put(Action("r"))
    q.put(Action("r"))
    q.put(Action("r"))
    q.put(Action("r"))
    
    q.put(Action("shoot", 600, 45))

        

if __name__ == "__main__":
    BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
    print(BASE_DIR)
    thread = threading.Thread(target = test)
    thread.start()
    window = PhysWin(500, 600)
    pyglet.app.run()
    thread._stop()
