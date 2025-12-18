"""
Pipe Balancing Game - A simple physics-based game where you balance a pipe
"""

import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (100, 149, 237)
GREEN = (34, 139, 34)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Physics
GRAVITY = 0.5
PIPE_LENGTH = 150
PIPE_WIDTH = 20
MAX_ANGLE = 45  # degrees
FRICTION = 0.98


class Pipe:
    """Represents the pipe that needs to be balanced"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # in degrees
        self.angular_velocity = 0  # degrees per frame
        self.length = PIPE_LENGTH
        self.width = PIPE_WIDTH
        
    def update(self, player_input):
        """Update pipe physics based on player input and gravity"""
        # Apply gravity-like torque (pipe wants to fall straight down)
        gravity_torque = math.sin(math.radians(self.angle)) * 0.3
        self.angular_velocity += gravity_torque
        
        # Apply player input
        if player_input == "left":
            self.angular_velocity -= 0.5
        elif player_input == "right":
            self.angular_velocity += 0.5
        
        # Apply friction
        self.angular_velocity *= FRICTION
        
        # Clamp angular velocity
        self.angular_velocity = max(min(self.angular_velocity, 5), -5)
        
        # Update angle
        self.angle += self.angular_velocity
        
        # Clamp angle to prevent flipping
        self.angle = max(min(self.angle, MAX_ANGLE), -MAX_ANGLE)
        
    def draw(self, surface):
        """Draw the pipe on the surface"""
        # Convert angle to radians
        angle_rad = math.radians(self.angle)
        
        # Calculate end point of pipe
        end_x = self.x + self.length * math.sin(angle_rad)
        end_y = self.y - self.length * math.cos(angle_rad)
        
        # Draw pipe
        pygame.draw.line(surface, BLUE, (self.x, self.y), (end_x, end_y), self.width)
        
        # Draw pivot point (fulcrum)
        pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), 8)
        
        # Draw angle indicator
        indicator_text = f"Angle: {self.angle:.1f}°"
        return indicator_text
    
    def is_balanced(self):
        """Check if pipe is reasonably balanced"""
        return abs(self.angle) < 10  # Within 5 degrees


class Game:
    """Main game class"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Pipe Balancing Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.pipe = Pipe(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self.score = 0
        self.time = 0
        self.game_over = False
        self.balanced_time = 0
        
    def handle_input(self):
        """Handle player input"""
        player_input = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE and self.game_over:
                    self.__init__()  # Reset game
        
        # Continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            player_input = "left"
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            player_input = "right"
        
        return True, player_input
    
    def update(self, player_input):
        """Update game state"""
        if self.game_over:
            return
        
        self.pipe.update(player_input)
        self.time += 1
        
        # Check if balanced
        if self.pipe.is_balanced():
            self.balanced_time += 1
            self.score = int(self.balanced_time / FPS)  # Convert frames to seconds
        else:
            self.balanced_time = 0
        
        # Game over if angle exceeds max
        if abs(self.pipe.angle) >= MAX_ANGLE:
            self.game_over = True
    
    def draw(self):
        """Draw game elements"""
        self.screen.fill(WHITE)
        
        # Draw ground/floor
        pygame.draw.line(self.screen, GRAY, (0, WINDOW_HEIGHT - 50), 
                        (WINDOW_WIDTH, WINDOW_HEIGHT - 50), 3)
        
        # Draw pipe
        angle_text = self.pipe.draw(self.screen)
        
        # Draw UI
        score_text = self.font.render(f"Balance Time: {self.score}s", True, BLACK)
        self.screen.blit(score_text, (20, 20))
        
        angle_surface = self.small_font.render(angle_text, True, BLACK)
        self.screen.blit(angle_surface, (20, 60))
        
        # Draw balance indicator
        if self.pipe.is_balanced():
            balanced_text = self.small_font.render(" BALANCED!", True, GREEN)
        else:
            balanced_text = self.small_font.render("Keep it balanced!", True, RED)
        self.screen.blit(balanced_text, (20, 100))
        
        # Draw controls
        controls = [
            "Controls: LEFT/RIGHT or A/D to balance",
            "SPACE to restart | ESC to quit"
        ]
        for i, control in enumerate(controls):
            control_surface = self.small_font.render(control, True, GRAY)
            self.screen.blit(control_surface, (20, WINDOW_HEIGHT - 80 + i * 30))
        
        # Draw game over screen
        if self.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = self.font.render("GAME OVER!", True, RED)
            final_score_text = self.font.render(f"Final Time: {self.score}s", True, WHITE)
            restart_text = self.small_font.render("Press SPACE to restart", True, YELLOW)
            
            self.screen.blit(game_over_text, 
                           (WINDOW_WIDTH // 2 - game_over_text.get_width() // 2, 200))
            self.screen.blit(final_score_text, 
                           (WINDOW_WIDTH // 2 - final_score_text.get_width() // 2, 270))
            self.screen.blit(restart_text, 
                           (WINDOW_WIDTH // 2 - restart_text.get_width() // 2, 350))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            result = self.handle_input()
            if isinstance(result, bool):
                running = result
            else:
                running, player_input = result
            
            self.update(player_input)
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()
