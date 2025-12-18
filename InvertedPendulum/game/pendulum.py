# Inverted Pendulum Game - A physics simulation game where the player balances a pendulum on a moving cart
# Import required libraries for game rendering (pygame) and numerical computations (numpy)
import pygame, sys
import numpy as np
from pygame.locals import *

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
        
        # Initialize the pendulum state
        self.reset_state()

    def reset_state(self):
        """Initializes pendulum in upright state with small perturbation"""
        self.is_dead = False  # Tracks if the pendulum has fallen (game over)
        self.time = 0  # Time elapsed in game frames
        self.x_cart = self.WINDOWWIDTH / 2  # Cart's initial x position (center of window)
        self.v_cart = 0  # Cart's initial velocity
        # Angle of pendulum (theta = 0 upright, omega positive into the screen)
        # Small random perturbation to make the game more challenging
        self.theta = np.random.uniform(-0.01, 0.01)
        self.omega = 0  # Initial angular velocity of pendulum

    def get_state(self):
        """Returns the current state of the pendulum system as a tuple"""
        return (self.is_dead, self.time, self.x_cart,
                self.v_cart, self.theta, self.omega)

    def set_state(self, state):
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
            raise RuntimeError("tried to call update_state while state was dead")
        
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
        self.theta += self.omega + self.v_cart * np.cos(self.theta) / float(self.PENDULUMLENGTH)
        
        # Update angular velocity based on gravity and pendulum angle
        self.omega += self.GRAVITY * np.sin(self.theta) / float(self.PENDULUMLENGTH)
        
        # Apply cart acceleration based on player action
        if action == "Left":
            self.v_cart -= self.A_CART
        elif action == "Right":
            self.v_cart += self.A_CART
        elif action == "None":
            self.v_cart = 0
        else:
            raise RuntimeError("action must be 'Left', 'Right', or 'None'")
        
        # Check if pendulum has fallen (angle exceeds 90 degrees)
        if abs(self.theta) >= np.pi / 2:
            self.is_dead = True
        
        return self.time, self.x_cart, self.v_cart, self.theta, self.omega


# InvertedPendulumGame class: Handles rendering and game loop for the inverted pendulum game
class InvertedPendulumGame(object):
    def __init__(self, windowdims, cartdims, penddims,
                 gravity, a_cart, refreshfreq, pendulum = None):
        # Use provided pendulum instance or create a new one
        if pendulum is None:
            self.pendulum = InvertedPendulum(windowdims, cartdims, penddims, gravity, a_cart)
        else:
            self.pendulum = pendulum
        
        # Store window dimensions
        self.WINDOWWIDTH = windowdims[0]
        self.WINDOWHEIGHT = windowdims[1]

        # Store cart dimensions
        self.CARTWIDTH = cartdims[0]
        self.CARTHEIGHT = cartdims[1]
        
        # Store pendulum dimensions
        self.PENDULUMWIDTH = penddims[0]
        self.PENDULUMLENGTH = penddims[1]

        # Get cart y-position from pendulum object
        self.Y_CART = self.pendulum.Y_CART
        
        # Initialize game state variables
        self.time = 0  # Time gives time in frames
        self.high_score = 0  # Track the best score achieved
        
        # Initialize pygame and create game window
        pygame.init()
        self.clock = pygame.time.Clock()
        
        # Store refresh frequency (frames per second for state updates)
        self.REFRESHFREQ = refreshfreq
        
        # Create pygame display surface
        self.surface = pygame.display.set_mode(windowdims, 0, 32)
        pygame.display.set_caption('Inverted Pendulum Game')
        
        # Pre-compute array specifying corners of pendulum to be drawn
        # This represents the pendulum shape in local coordinates before rotation
        self.static_pendulum_array = np.array(
            [[-self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH],
             [-self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH]]).T
        
        # Define color constants (RGB)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

    def draw_cart(self, x, theta):
        """
        Renders the cart and pendulum to the game window.
        
        Args:
            x: X-coordinate of the cart's center
            theta: Current angle of the pendulum (in radians)
        """
        # Draw the cart as a black rectangle
        cart = pygame.Rect(x - self.CARTWIDTH // 2, self.Y_CART, self.CARTWIDTH, self.CARTHEIGHT)
        pygame.draw.rect(self.surface, self.BLACK, cart)
        
        # Transform pendulum coordinates: rotate by theta angle, then translate to cart position
        pendulum_array = np.dot(self.rotation_matrix(theta), self.static_pendulum_array)
        pendulum_array += np.array([[x], [self.Y_CART]])
        
        # Draw the pendulum as a black polygon
        pendulum = pygame.draw.polygon(self.surface, self.BLACK,
            ((pendulum_array[0, 0], pendulum_array[1, 0]),
             (pendulum_array[0, 1], pendulum_array[1, 1]),
             (pendulum_array[0, 2], pendulum_array[1, 2]),
             (pendulum_array[0, 3], pendulum_array[1, 3])))

    @staticmethod
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

    def render_text(self, text, point, position = "center", fontsize = 48):
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

    def starting_page(self):
        """Displays the game's starting/title screen"""
        self.surface.fill(self.WHITE)
        
        # Display game title
        self.render_text("Inverted Pendulum",
                         (0.5 * self.WINDOWWIDTH, 0.4 * self.WINDOWHEIGHT))
        
        # Display creator credit
        self.render_text("A Game by Adam Strandberg",
                         (0.5 * self.WINDOWWIDTH, 0.5 * self.WINDOWHEIGHT),
                         fontsize = 30)
        
        # Display start instruction
        self.render_text("Press Enter to Begin",
                         (0.5 * self.WINDOWWIDTH, 0.7 * self.WINDOWHEIGHT),
                         fontsize = 30)
        
        # Update the display
        pygame.display.update()

    def game_round(self):
        """
        Runs a single game round where the player balances the pendulum.
        Continues until the pendulum falls (is_dead becomes True).
        """
        # Reset pendulum to initial state
        self.pendulum.reset_state()
        
        # Initialize action to no movement
        action = "None"
        
        # Main game loop: runs until pendulum falls
        while not self.pendulum.is_dead:
            # Handle user input events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Detect key presses
                if event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        action = "Left"
                    if event.key == K_RIGHT:
                        action = "Right"
                
                # Detect key releases
                if event.type == KEYUP:
                    if event.key == K_LEFT:
                        action = "None"
                    if event.key == K_RIGHT:
                        action = "None"
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            
            # Update game physics with player's action
            t, x, _, theta, _ = self.pendulum.update_state(action)
            self.time = t    
            
            # Clear screen with white background
            self.surface.fill(self.WHITE)
            
            # Draw cart and pendulum
            self.draw_cart(x, theta)

            # Display elapsed time
            time_text = "t = {}".format(self.time_seconds())
            self.render_text(time_text, (0.1 * self.WINDOWWIDTH, 0.1 * self.WINDOWHEIGHT),
                             position = "topleft", fontsize = 40)
            
            # Update display
            pygame.display.update()
            
            # Maintain constant frame rate
            self.clock.tick(self.REFRESHFREQ)
        
        # Update high score if current score is better
        if (self.time_seconds()) > self.high_score:
            self.high_score = self.time_seconds()

    def end_of_round(self):
        """
        Displays the end-of-round screen showing the score and high score.
        Waits for player input to continue or exit.
        """
        self.surface.fill(self.WHITE)
        
        # Draw the final state of cart and pendulum
        self.draw_cart(self.pendulum.x_cart, self.pendulum.theta)
        
        # Display current round score
        self.render_text("Score: {}".format(self.time_seconds()),
                         (0.5 * self.WINDOWWIDTH, 0.3 * self.WINDOWHEIGHT))
        
        # Display best score achieved so far
        self.render_text("High Score : {}".format(self.high_score),
                         (0.5 * self.WINDOWWIDTH, 0.4 * self.WINDOWHEIGHT))
        
        # Display player instructions
        self.render_text("(Enter to play again, ESC to exit)",
                         (0.5 * self.WINDOWWIDTH, 0.85 * self.WINDOWHEIGHT),
                         fontsize = 30)
        
        # Update display
        pygame.display.update()

    def game(self):
        """
        Main game loop that orchestrates the game flow.
        Displays starting page, runs game rounds, and handles game over conditions.
        """
        while True:
            # Handle events during menu/between rounds
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                
                # Run a single game round and display results
                self.game_round()
                self.end_of_round()


# Entry point for the game
def main():
    """Initializes and starts the inverted pendulum game"""
    inv = InvertedPendulumGame(WINDOWDIMS, CARTDIMS, PENDULUMDIMS, GRAVITY, A_CART, REFRESHFREQ)
    inv.game()

# Run the game when this script is executed directly
if __name__ == '__main__':
    main()