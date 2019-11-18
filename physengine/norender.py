import pymunk
from math import cos, sin, radians
import random


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
        self.COORD_STEP_SIZE = 50
        #self.MIN_NET_X = self.winwidth / 2 + self.COORD_STEP_SIZE
        #self.MAX_NET_X = self.winwidth - 50
        self.MIN_NET_X = self.winwidth * (3/4)
        self.MAX_NET_X = self.MIN_NET_X # For now, don't allow net to vary in x direction
        self.MIN_NET_Y = 200
        self.MAX_NET_Y = self.winheight - 100
        
        self.MIN_BALL_X = 0 + self.COORD_STEP_SIZE
        self.MAX_BALL_X = self.winwidth / 2 - self.COORD_STEP_SIZE
        self.STARTING_BALL_Y = 200
        
        self.MIN_SHOT_VEL = 100
        self.MAX_SHOT_VEL = 1000
        self.VELOCITY_STEP_SIZE = 20
        
        self.MIN_SHOT_ANGLE = 0
        self.MAX_SHOT_ANGLE = 90
        self.ANGLE_STEP_SIZE = 5
        
        self.WIND_FORCE = 200
        
        self.DEFAULT_GRAVITY = -900
        
        self.MAX_STEPS = 250
        
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
        self.steps_taken = 0
        self.ball_missed_by = 1
        
        if doRandomize:
            self.initial_basketball_x = random.randrange(self.MIN_BALL_X, self.MAX_BALL_X, self.COORD_STEP_SIZE)
            #self.initial_basketball_x = self.MIN_BALL_X
            # Basketball will always start at the same height
            self.initial_basketball_y = self.STARTING_BALL_Y
            
            if self.MIN_NET_X == self.MAX_NET_X:
                self.initial_hoop_x = self.MIN_NET_X
            else:
                self.initial_hoop_x = random.randrange(self.MIN_NET_X, self.MAX_NET_X, self.COORD_STEP_SIZE)
            self.initial_hoop_y = random.randrange(self.MIN_NET_Y, self.MAX_NET_Y, self.COORD_STEP_SIZE)
            
            wind_angle = random.randrange(0, 360, 15)
            #wind_angle = 180
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
        vel_x = velocity * cos(radians(angle))
        vel_y = velocity * sin(radians(angle))
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

        
    #def check_bounds(self):
    #    if(self.basketball_body.position.x > self.rim1_body.position.x and self.basketball_body.position.x < self.rim2_body.position.x 
    #        and self.basketball_body.position.y < self.rim1_body.position.y + 10 and self.basketball_body.position.y > self.rim1_body.position.y - 10) and self.basketball_body.velocity.y < 0:
    #        self.ball_hit = True
            
    def check_bounds(self):
        if self.basketball_body.position.y < self.rim1_body.position.y + 10 and self.basketball_body.position.y > self.rim1_body.position.y - 10 and self.basketball_body.velocity.y < 0:
            if self.basketball_body.position.x > self.rim1_body.position.x and self.basketball_body.position.x < self.rim2_body.position.x:
                self.ball_hit = True
            else:
                self.ball_missed_by = self.normalize(abs(self.basketball_body.position.x - self.rim1_body.position.x))
                
    def normalize(self, dist):
        # Based on normalization formula: (x - xmin)/(xmax - xmin); since xmin is 0, it is simply x/xmax. For simplicity, let xmax be winwidth.
        return dist/self.winwidth
        
    def getMissedBy(self):
        return self.ball_missed_by
            
    def shoot(self, action):
        velocity = action.velocity
        angle = action.angle
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
        if (posx >= self.MAX_BALL_X and dir == "r") or (posx <= self.MIN_BALL_X and dir == "l"):
            return

        self.space.remove(self.basketball_body)
        self.space.remove(self.basketball_shape)
        
        if dir == "r":
            self.basketball_body.position = posx + self.COORD_STEP_SIZE, posy
        else:
            self.basketball_body.position = posx - self.COORD_STEP_SIZE, posy

        self.space.add(self.basketball_body)
        self.space.add(self.basketball_shape)    
            
    def getCoords(self):
        return int(self.netx), int(self.nety), int(self.basketball_body.position.x)
        
    def getStateSpace(self):
        # Need to add 1 to each of these; without adding +1, it only represents the maximum possible index, not the actual size of the dimension
        return [
            int((self.MAX_NET_X - self.MIN_NET_X) / self.COORD_STEP_SIZE) + 1, # net_x - This is how many starting spots available for x coord of net
            int((self.MAX_NET_Y - self.MIN_NET_Y) / self.COORD_STEP_SIZE) + 1, # net_y - This is how many starting spots available for y coord of net
            int((self.MAX_BALL_X - self.MIN_BALL_X) / self.COORD_STEP_SIZE) + 1 # ball_x - This is how many starting spots available for ball; no variance in ball starting y is needed
        ]
        
    # This function is a bit weird and works differently than getStateSpace
    # The state space is completely multiplicative - for each m states in net_x, there are n states in net_y, and for each
    # of those n states in net_y, there's l ball-x states.
    # Equivalently, in getActionSpace, there would be 3 actions (left, right and shoot), and for each of those a set of velocity states,
    # and for each of those velocities a set of angle states. This is not true though, as only shoot should have velocities and angles
    # Therefore, to reduce the memory requirement, this list will be one dimensional; 
    # the first two indexes will be for left/right, and the rest of the
    # list will be for shoot/velocity/angles (these will be mapped to a unique index)
    def getActionSpace(self):
        return int(2 + ((self.MAX_SHOT_VEL - self.MIN_SHOT_VEL) / self.VELOCITY_STEP_SIZE) * ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE)) + 1
            
    def mapValsToState(self, net_x, net_y, ball_x):
        #print("net_x: ", net_x, ", net_y: ", net_y, ", ball_x: ", ball_x)
        #print("net_x_idx: ", (net_x - self.MIN_NET_X)/self.COORD_STEP_SIZE, ", net_y_idx: ", (net_y - self.MIN_NET_Y)/self.COORD_STEP_SIZE, ", ball_x_idx: ", (ball_x - self.MIN_BALL_X)/self.COORD_STEP_SIZE)
        return int((net_x - self.MIN_NET_X)/self.COORD_STEP_SIZE), int((net_y - self.MIN_NET_Y)/self.COORD_STEP_SIZE), int((ball_x - self.MIN_BALL_X)/self.COORD_STEP_SIZE)
        
    def mapValsToAction(self, action):
        if action.action_type == "l":
            return 0
        elif action.action_type == "r":
            return 1
        else:
            # Always add 2, because 0 and 1 are reserved for left and right
            velocityidx = ((action.velocity - self.MIN_SHOT_VEL) / self.VELOCITY_STEP_SIZE) * ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE)
            angleidx = ((action.angle - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE)
            return int(2 + velocityidx + angleidx)
            
    def mapActionToVals(self, actionidx):
        if actionidx == 0:
            return Action("l", 0, 0)
        elif actionidx == 1:
            return Action("r", 0, 0)
        else:
            actionidx -= 2 # first get rid of the left and right actions
            angleval = actionidx % ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE)
            actionidx -= angleval # subtract out the angle index to get the velocity index
            velocityval = actionidx / ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE)
            velocityval = velocityval * self.VELOCITY_STEP_SIZE + self.MIN_SHOT_VEL
            angleval = (angleval * self.ANGLE_STEP_SIZE) + self.MIN_SHOT_ANGLE
            return Action("shoot", velocityval, angleval)

        
class PymunkSpaceNoRender(PymunkSpace):
    def __init__ (self, winwidth, winheight):
        PymunkSpace.__init__(self, winwidth, winheight)
        
    def shoot(self, action, dt):
        super().shoot(action)
        while (not self.ball_hit) and (not self.ball_missed):
            self.no_render_update(dt)
            self.steps_taken += 1
            # Make sure the ball isn't stuck
            if self.steps_taken > self.MAX_STEPS:
                self.reset_space(False)
                return False
        if self.ball_hit:
            # If the ball hits, reset the space and randomize ball and hoop locations
            self.reset_space(True)
            return 0
        else:
            # If the ball misses, reset to the starting locations but don't randomize locations
            missed_by = self.ball_missed_by
            self.reset_space(False)
            return missed_by
        
    def no_render_update(self, dt):
        self.space.step(dt)
        
        self.check_bounds()


if __name__ == "__main__":
    pmSpace = PymunkSpaceNoRender(500, 600)

    while True:
        pmSpace.move("r")
        pmSpace.move("r")
        if pmSpace.shoot(Action("shoot", 300, 45), 0.02):
            print ("Basket hit!")
        else:
            print ("Basket missed!")
