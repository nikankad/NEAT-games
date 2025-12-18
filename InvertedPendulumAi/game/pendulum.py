# Inverted Pendulum Game - A physics simulation game where the player balances a pendulum on a moving cart
# Import required libraries for game rendering (pygame) and numerical computations (numpy)
import pygame
import sys
import numpy as np
from pygame.locals import *
import neat
# import utils.visualize


# Configuration constants for the game
WINDOWDIMS = (1200, 600)  # Window width and height in pixels
CARTDIMS = (50, 10)  # Cart width and height in pixels
PENDULUMDIMS = (6, 200)  # Pendulum width and length in pixels
GRAVITY = 0.13  # Gravity acceleration factor for the pendulum
REFRESHFREQ = 100  # Game refresh frequency (frames per second)
A_CART = 0.15  # Cart acceleration magnitude when moving left or right
# InvertedPendulum class: Handles the physics simulation of the inverted pendulum system
class InvertedPendulum(object):
    def __init__(self, windowdims, cartdims, penddims, gravity, a_cart):
        # Store window dimensions
        self.WINDOWWIDTH = windowdims[0]
        self.WINDOWHEIGHT = windowdims[1]
        # Store cart dimensions
        self.CARTWIDTH = cartdims[0]
        self.CARTHEIGHT = cartdims[1]

        # Store pendulum dimensions
        self.PENDULUMWIDTH = penddims[0]
        self.PENDULUMLENGTH = penddims[1]

        # Store physics parameters
        self.GRAVITY = gravity
        self.A_CART = a_cart

        # Calculate the y-coordinate of the cart (3/4 down the window)
        self.Y_CART = 3 * self.WINDOWHEIGHT / 4
        self.color = (np.random.randint(0, 256), np.random.randint(
            0, 256), np.random.randint(0, 256))

        # Initialize the pendulum state
        self.reset_state()

    def reset_state(self):
        """Initializes pendulum in upright state with small perturbation"""
        self.is_dead = False  # Tracks if the pendulum has fallen (game over)
        self.time = 0  # Time elapsed in game frames
        # Cart's initial x position (center of window)
        self.x_cart = np.random.uniform(0.4, 0.6) * self.WINDOWWIDTH
        self.v_cart = np.random.uniform(-1, 1)        # Angle of pendulum (theta = 0 upright, omega positive into the screen)
        # Small random perturbation to make the game more challenging
        self.theta =np.random.uniform(-np.pi/4, np.pi/4) 
        self.omega = 0  # Initial angular velocity of pendulum

    def get_state(self):
        """Returns the current state of the pendulum system as a tuple"""
        return (self.is_dead, self.time, self.x_cart,
                self.v_cart, self.theta, self.omega)

    def set_state(self, state):
        """
        Sets the pendulum system state from a tuple.

        Args:
            state (tuple): A tuple containing six elements in the following order:
                - is_dead (bool): Whether the pendulum system has failed/reached terminal state
                - t (float): Current time value
                - x (float): Cart position
                - v (float): Cart velocity
                - theta (float): Pendulum angle (in radians)
                - omega (float): Pendulum angular velocity

        Returns:
            None
        """
        """Sets the pendulum system state from a tuple"""
        is_dead, t, x, v, theta, omega = state
        self.is_dead = is_dead
        self.time = t
        self.x_cart = x
        self.v_cart = v
        self.theta = theta
        self.omega = omega

    def update_state(self, action):
        """
        Updates the pendulum system state based on physics and player action.
        All the physics calculations are performed here.

        Args:
            action: String indicating cart movement ("Left", "Right", or "None")

        Returns:
            Tuple of (time, x_cart, v_cart, theta, omega)
        """
        assert isinstance(action, str)

        # Check if pendulum is already fallen
        if self.is_dead:
            raise RuntimeError(
                "tried to call update_state while state was dead")

        # Increment time by one frame
        self.time += 1

        # Update cart position based on velocity
        self.x_cart += self.v_cart

        # Boundary conditions: cart stops when it hits the walls
        if self.x_cart <= self.CARTWIDTH / 2:
            self.x_cart = self.CARTWIDTH / 2
            self.v_cart = 0
        elif self.x_cart >= self.WINDOWWIDTH - self.CARTWIDTH / 2:
            self.x_cart = self.WINDOWWIDTH - self.CARTWIDTH / 2
            self.v_cart = 0

        # Update pendulum angle: term from angular velocity + term from motion of cart
        self.theta += self.omega + self.v_cart * \
            np.cos(self.theta) / float(self.PENDULUMLENGTH)

        # Update angular velocity based on gravity and pendulum angle
        self.omega += self.GRAVITY * \
            np.sin(self.theta) / float(self.PENDULUMLENGTH)

        # Apply cart acceleration based on player action
        if action == "Left":
            self.v_cart -= self.A_CART
        elif action == "Right":
            self.v_cart += self.A_CART
        elif action == "None":
            self.v_cart *= 0.07
        else:
            raise RuntimeError("action must be 'Left', 'Right', or 'None'")

        # Check if pendulum has fallen (angle exceeds 90 degrees)
        if abs(self.theta) >= np.pi / 2:
            self.is_dead = True

        return self.time, self.x_cart, self.v_cart, self.theta, self.omega


    def render_text(self, text, point, position="center", fontsize=48):
        """
        Renders text on the game window at the specified position.

        Args:
            text: String to render
            point: Tuple of (x, y) coordinates
            position: Positioning mode - "center" or "topleft" (default: "center")
            fontsize: Font size in pixels (default: 48)
        """
        font = pygame.font.SysFont(None, fontsize)
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        text_rect = text_render.get_rect()

        # Position text based on the specified alignment
        if position == "center":
            text_rect.center = point
        elif position == "topleft":
            text_rect.topleft = point

        # Draw text to the game surface
        self.surface.blit(text_render, text_rect)

    def time_seconds(self):
        """Converts elapsed game frames to seconds"""
        return self.time / float(self.REFRESHFREQ)


def rotation_matrix(theta):
    """
    Creates a 2D rotation matrix for the given angle.

    Args:
        theta: Angle in radians

    Returns:
        2x2 rotation matrix
    """
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-1 * np.sin(theta), np.cos(theta)]])


def run_pendulum(genomes, config):
    nets = []
    ge = []
    pendulums = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        pendulum = InvertedPendulum(
            WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART)
        g.fitness = 0
        nets.append(net)
        pendulums.append(pendulum)
        ge.append(g)

    # Initialize game state variables
    high_score = 0  # Track the best score achieved

    score = 0
   
    # Initialize pygame and create game window
    pygame.init()
    clock = pygame.time.Clock()

    # Store refresh frequency (frames per second for state updates)

    # Create pygame display surface
    surface = pygame.display.set_mode(WINDOWDIMS, 0, 32)
    pygame.display.set_caption('Inverted Pendulum Game')


    # Define color constants (RGB)

    while pendulums and ge:
        clock.tick(REFRESHFREQ)

        surface.fill((0,0,0))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # fitness is based on how long we are alive and based on the angle of the pendulum
        

        for i, pendulum in enumerate(pendulums):
            if pendulum.is_dead:
                continue
            is_dead, time, x, v, theta, omega = pendulum.get_state()
            inputs =[(x - WINDOWDIMS[0]/2) / (WINDOWDIMS[0]/2),
            v / 5.0,
            theta / (np.pi / 2),
            omega / 5.0]

            output = nets[i].activate(inputs)
            action = ["Left", "None", "Right"][np.argmax(output)]
            
            # UPDATE the physics
            pendulum.update_state(action)
            
            # Pre-compute array specifying corners of pendulum to be drawn
            # This represents the pendulum shape in local coordinates before rotation
            # now we have to draw our carts
            static_pendulum_array = np.array(
                [[PENDULUMDIMS[0] / 2, 0],
                 [PENDULUMDIMS[0] / 2, 0],
                 [PENDULUMDIMS[0] / 2, -PENDULUMDIMS[1]],
                 [-PENDULUMDIMS[0] / 2, -PENDULUMDIMS[1]]]).T
            # Draw the cart as a black rectangle
            cart = pygame.Rect(x - CARTDIMS[0] // 2,
                               pendulum.Y_CART, CARTDIMS[0], CARTDIMS[1])
            pygame.draw.rect(surface, pendulum.color, cart)

            # Transform pendulum coordinates: rotate by theta angle, then translate to cart position
            pendulum_array = np.dot(rotation_matrix(theta), static_pendulum_array)
            pendulum_array += np.array([[x], [pendulum.Y_CART]])

            # Draw the pendulum as a black polygon
            pygame.draw.polygon(surface, pendulum.color,
                                ((pendulum_array[0, 0], pendulum_array[1, 0]),
                                 (pendulum_array[0, 1], pendulum_array[1, 1]),
                                 (pendulum_array[0, 2], pendulum_array[1, 2]),
                                 (pendulum_array[0, 3], pendulum_array[1, 3])))
            ge[i].fitness += 1
            ge[i].fitness -= abs(theta) * 2
            ge[i].fitness -= abs(x - WINDOWDIMS[0]/2) / 200
            
        pygame.display.update()
        for i in reversed(range(len(pendulums))):
                if pendulums[i].is_dead:
                    ge[i].fitness -= 20
                    ge.pop(i)
                    nets.pop(i)
                    pendulums.pop(i)
        clock.tick(REFRESHFREQ)

                    
                



if __name__ == "__main__":
    # Set configuration file
    config_path = "InvertedPendulumAi\config-inverted-pendulum.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(run_pendulum,  10)
    print("Best fitness:", winner.fitness)
