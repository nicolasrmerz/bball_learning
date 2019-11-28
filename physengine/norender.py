from modules import pymunk
from math import cos, sin, radians
import random

# The action class, which should have "l", "r", or "shoot" for action_type, and a velocity and angle when action_type="shoot"
class Action():
    def __init__(self, action_type, velocity=0, angle=0):
        self.action_type = action_type
        self.velocity = velocity
        self.angle = angle
        
# This class defines the underlying pymunk engine, which does all physics simulations
class PymunkSpace():
    def __init__(self, cfg):
        self.winwidth = int(cfg['winwidth'])
        self.winheight = int(cfg['winheight'])
        # Set constraints of the system using the "Game" part of the config.
        # Minimum/maximum net and ball positions are scaled to the window size
        self.COORD_STEP_SIZE = int(cfg['CoordStepSize'])
        self.MIN_NET_X = self.winwidth / 2 + self.COORD_STEP_SIZE
        self.MAX_NET_X = self.winwidth - self.COORD_STEP_SIZE
        self.MIN_NET_Y = 4 * self.COORD_STEP_SIZE
        self.MAX_NET_Y = self.winheight - 100
        
        self.MIN_BALL_X = 0 + self.COORD_STEP_SIZE
        self.MAX_BALL_X = self.winwidth / 2 - self.COORD_STEP_SIZE
        self.STARTING_BALL_Y = 4 * self.COORD_STEP_SIZE
        
        self.MIN_SHOT_VEL = int(cfg['MinShotVel'])
        self.MAX_SHOT_VEL = int(cfg['MaxShotVel'])
        self.VELOCITY_STEP_SIZE = int(cfg['VelocityStepSize'])
        
        self.MIN_SHOT_ANGLE = int(cfg['MinShotAngle'])
        self.MAX_SHOT_ANGLE = int(cfg['MaxShotAngle'])
        self.ANGLE_STEP_SIZE = int(cfg['AngleStepSize'])
        
        self.WIND_FORCE = int(cfg['WindForce'])
        
        self.DEFAULT_GRAVITY = int(cfg['DefaultGravity'])
        
        # This is the number of steps the engine will simulate before it resets itself, to make sure the ball does not get stuck
        self.MAX_STEPS = 250
        
        # Which wind model to use
        self.ALWAYS_RANDOMIZE_WIND = cfg.getboolean('AlwaysRandomizeWind')
        
        self.reset_space(True)
        
    # Wrapper for creating a circle in pymunk
    def create_circle(self, mass, radius, posx, posy, label, body_type):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment, body_type)
        #Initially start with basketball being kinetic, as we want to be able to freely control it before taking the shot
        shape = pymunk.Circle(body, radius)
        body.position = posx, posy
        # Just use these values for all objects; adjust if needed
        shape.elasticity = 0.95
        shape.friction = 0.5
        # Used to see if there's a collision between the basketball and the ground (shot was missed)
        shape.id = label
        return body, shape
        
    # Wrapper for creating a segment in pymunk
    def create_segment(self, p1x, p1y, p2x, p2y, label, thickness):
        body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (p1x, p1y), (p2x, p2y), thickness)
        shape.id = "connector"
        shape.friction = 0.5
        shape.elasticity = 0.95
        return body, shape
    
    # This resets the space to new starting ball and net positions
    def reset_space(self, doRandomize):
        self.space = pymunk.Space()
        # These are the handlers for handling the collision - the intent is to reset the space when pymunk detects a collision between the ball and the ground
        self.handler = self.space.add_default_collision_handler();
        self.handler.begin = self.coll_begin
        self.handler.pre_solve = self.coll_pre
        self.handler.post_solve = self.coll_post
        self.handler.separate = self.coll_separate
        # These flags keep track of whether a basket was made or missed
        self.ball_missed = False
        self.ball_hit = False
                
        # Reset the step counter
        self.steps_taken = 0

        # If the proportional reward scheme is used, this tracks how close the agent was to making the shot
        self.ball_missed_by = 1
        
        # If doRandomize is passed, re-randomize the ball and net starting positions.\
        # Generally, doRandomize is true when a basket is made, and false when the basket is missed.
        if doRandomize:
            if self.MIN_BALL_X == self.MAX_BALL_X:
                self.initial_basketball_x = self.MIN_BALL_X
            else:
                self.initial_basketball_x = random.randrange(self.MIN_BALL_X, self.MAX_BALL_X, self.COORD_STEP_SIZE)
            #self.initial_basketball_x = self.MIN_BALL_X
            # Basketball will always start at the same height
            self.initial_basketball_y = self.STARTING_BALL_Y
            
            if self.MIN_NET_X == self.MAX_NET_X:
                self.initial_hoop_x = self.MIN_NET_X
            else:
                self.initial_hoop_x = random.randrange(self.MIN_NET_X, self.MAX_NET_X, self.COORD_STEP_SIZE)
            
            if self.MIN_NET_Y == self.MAX_NET_Y:
                self.initial_hoop_y = self.MIN_NET_Y
            else:
                self.initial_hoop_y = random.randrange(self.MIN_NET_Y, self.MAX_NET_Y, self.COORD_STEP_SIZE)
                
            # If wind is not always randomized, randomize it here when the environment does a hard, re-randomized reset
            if not self.ALWAYS_RANDOMIZE_WIND:
                wind_angle = random.randrange(0, 360, 60)
                self.wind_x, self.wind_y = self.get_x_y(self.WIND_FORCE, wind_angle)
            
        # If wind is always randomized, re-randomize the wind
        if self.ALWAYS_RANDOMIZE_WIND:
            wind_angle = random.randrange(0, 360, 60)
            self.wind_x, self.wind_y = self.get_x_y(self.WIND_FORCE, wind_angle)

        # Reset the gravity by setting it to the default and adding the wind force
        self.space.gravity = 0 + self.wind_x, self.DEFAULT_GRAVITY + self.wind_y

        # Re-add all the shapes required
        self.mass = 1
        self.ball_radius = 15

        # The basketball is intially set to be a static body, so it does not move anywhere until it is explicitly shot
        self.basketball_body, self.basketball_shape = self.create_circle(self.mass, self.ball_radius, 
            self.initial_basketball_x, self.initial_basketball_y, "basketball", pymunk.Body.STATIC)
        
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_shape = pymunk.Poly.create_box(self.ground_body, size=(10000, 2 * self.COORD_STEP_SIZE))
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
            
            
    # Collision handlers for pymunk
    def coll_begin(self, arbiter, space, data):
        # Checks if the collision was between basketball and ground
        if arbiter.shapes[0].id == "basketball" and arbiter.shapes[1].id == "ground":
            self.ball_missed = True
        return True
        
    # To satisfy pymunk's complaining about not providing every single part of a proper reward handler...
    def coll_pre(self, arbiter, space, data):
        return True
        
    def coll_post(self, arbiter, space, data):
        pass
        
    def coll_separate(self, arbiter, space, data):
        pass
    
    # Convert a velocity and angle into x and y components
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
            
    # This check sees if the ball is within a small rectangle around the net
    # (more specifically, within the x coordinates of the two rims, within a threshold above or below the two rims, and the ball having a negative velocity)
    def check_bounds(self):
        if self.basketball_body.position.y < self.rim1_body.position.y + 10 and self.basketball_body.position.y > self.rim1_body.position.y - 10 and self.basketball_body.velocity.y < 0:
            if self.basketball_body.position.x > self.rim1_body.position.x and self.basketball_body.position.x < self.rim2_body.position.x:
                self.ball_hit = True
            else:
                self.ball_missed_by = self.normalize(abs(self.basketball_body.position.x - self.rim1_body.position.x))
                
    # Based on normalization formula: (x - xmin)/(xmax - xmin); since xmin is 0, it is simply x/xmax. For simplicity, let xmax be winwidth.
    def normalize(self, dist):
        return dist/self.winwidth
    
    # Return the missed_by value
    def getMissedBy(self):
        return self.ball_missed_by
    
    # Convert the ball into a Dynamic object, which obeys the laws of physics, and give it an initial velocity.
    # Pymunk didn't seem to like simply changing the body type, so instead the basketball was completely destroyed and recreated using
    # the x and y coordinates it had
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
        
    # Move the ball left or right COORD_STEP_SIZE units. Like with shoot, the ball is destroyed and recreated to satisfy pymunk's lack of desire to cooperate
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
            
    # Return the x coordinate of the ball (before it is shot, the y coordinate will never vary), and the x/y coordinates of the net
    def getCoords(self):
        return int(self.netx), int(self.nety), int(self.basketball_body.position.x)
        
    # Get the size of each of the dimensions of the state space
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
            
    # Convert the x coordinate of the ball and the x/y coordinates of the net into indices to access the state-action table
    def mapValsToState(self, net_x, net_y, ball_x):
        return int((net_x - self.MIN_NET_X)/self.COORD_STEP_SIZE), int((net_y - self.MIN_NET_Y)/self.COORD_STEP_SIZE), int((ball_x - self.MIN_BALL_X)/self.COORD_STEP_SIZE)
        
    # Convert a supplied action into its corresponding index in the action list
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
            
    # Convert an index from the action list into an actual action the agent can execute
    # Due to the fact the action list is 1-D, this is a little convoluted
    def mapActionToVals(self, actionidx):
        # Again, the first two indices are reserved for left and right
        if actionidx == 0:
            return Action("l", 0, 0)
        elif actionidx == 1:
            return Action("r", 0, 0)
        else:
            actionidx -= 2 # first get rid of the left and right actions
            angleval = actionidx % ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE) # Get an angle index - if the table were instead 3D, this would be the index for the angle dimension of the table
            actionidx -= angleval # subtract out the angle index
            velocityval = actionidx / ((self.MAX_SHOT_ANGLE - self.MIN_SHOT_ANGLE) / self.ANGLE_STEP_SIZE) # Get the velocity index
            velocityval = velocityval * self.VELOCITY_STEP_SIZE + self.MIN_SHOT_VEL # Convert the velocity into an actual value
            angleval = (angleval * self.ANGLE_STEP_SIZE) + self.MIN_SHOT_ANGLE # Convert the angle index into the actual value
            return Action("shoot", velocityval, angleval)

# This is an extension of the pymunk engine that is used when the agent is not rendered. When shoot is called here, it calls its parent version of shoot, then updates until the basket is actually made or missed
class PymunkSpaceNoRender(PymunkSpace):
    def __init__ (self, cfg):
        PymunkSpace.__init__(self, cfg)
        
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
