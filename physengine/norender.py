import render

class PymunkSpaceNoRender(render.PymunkSpace):
    def __init__ (self, winwidth, winheight):
        render.PymunkSpace.__init__(self, winwidth, winheight)
        
    def shoot(self, velocity, angle, dt):
        super().shoot(velocity, angle)
        while (not self.ball_hit) and (not self.ball_missed):
            self.no_render_update(dt)
        if self.ball_hit:
            # If the ball hits, reset the space and randomize ball and hoop locations
            self.reset_space(True)
            return True
        else:
            # If the ball misses, reset to the starting locations but don't randomize locations
            self.reset_space(False)
            return False
        
    def no_render_update(self, dt):
        self.space.step(dt)
        self.check_bounds()

pmSpace = PymunkSpaceNoRender(500, 600)

while True:
    pmSpace.move("r")
    pmSpace.move("r")
    if pmSpace.shoot(300, 45, 0.02):
        print ("Basket hit!")
    else:
        print ("Basket missed!")
