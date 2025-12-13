# Flappy Bird Game - Main Game Loop
import pygame
import random
import time
from pygame.locals import *
from game.model import NeatModel
# ==================== GAME CONSTANTS ====================
# Screen dimensions
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600

# Bird movement constants
SPEED = 20  # Flap strength
GRAVITY = 2.5  # Falling acceleration
GAME_SPEED = 15  # Speed of pipes and ground moving

# Ground dimensions
GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

# Pipe dimensions
PIPE_WIDHT = 80
PIPE_HEIGHT = 500

# Gap between upper and lower pipes
PIPE_GAP = 150

# fps
fps = 30

# Audio file paths
wing = 'game/assets/audio/wing.wav'  # Sound when bird flaps
hit = 'game/assets/audio/hit.wav'    # Sound when bird hits obstacle

# Initialize pygame mixer for sound effects
pygame.mixer.init()

# i need to have a dict of the flappy birds values vals: {PositionX, PositionY, Distance From next pipe }

# ==================== BIRD CLASS ====================


class Bird(pygame.sprite.Sprite):
    """Represents the flappy bird character"""

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        # Load bird animation frames (3 different wing positions)
        self.images = [pygame.image.load('game/assets/sprites/bluebird-upflap.png').convert_alpha(),
                       pygame.image.load(
                           'game/assets/sprites/bluebird-midflap.png').convert_alpha(),
                       pygame.image.load('game/assets/sprites/bluebird-downflap.png').convert_alpha()]

        # Initial vertical velocity
        self.speed = SPEED

        # Animation frame counter
        self.current_image = 0
        self.image = pygame.image.load(
            'game/assets/sprites/bluebird-upflap.png').convert_alpha()
        # Create collision mask from image
        self.mask = pygame.mask.from_surface(self.image)

        # Set initial bird position (left side, middle height)
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        """Update bird animation and apply gravity"""
        # Cycle through animation frames
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        # Apply gravity (increases falling speed)
        self.speed += GRAVITY

        # Update bird's vertical position
        self.rect[1] += self.speed

    def bump(self):
        """Make the bird flap (jump upward)"""
        self.speed = -SPEED

    def begin(self):
        """Animate bird during menu screen"""
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

# ==================== PIPE CLASS ====================


class Pipe(pygame.sprite.Sprite):
    """Represents a pipe obstacle"""

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        # Load pipe sprite
        self.image = pygame.image.load(
            'game/assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(
            self.image, (PIPE_WIDHT, PIPE_HEIGHT))

        # Set pipe position
        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        # Position pipe (inverted is top pipe, normal is bottom pipe)
        if inverted:
            # Flip image upside down for top pipe
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            # Position bottom pipe
            self.rect[1] = SCREEN_HEIGHT - ysize

        # Create collision mask
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """Move pipe to the left"""
        self.rect[0] -= GAME_SPEED


# ==================== GROUND CLASS ====================
class Ground(pygame.sprite.Sprite):
    """Represents the ground at the bottom of the screen"""

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        # Load ground sprite
        self.image = pygame.image.load(
            'game/assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(
            self.image, (GROUND_WIDHT, GROUND_HEIGHT))

        # Create collision mask
        self.mask = pygame.mask.from_surface(self.image)

        # Position ground at the bottom of the screen
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        """Move ground to the left"""
        self.rect[0] -= GAME_SPEED

# ==================== UTILITY FUNCTIONS ====================


def is_off_screen(sprite):
    """Check if a sprite has moved off the left side of the screen"""
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    """Generate a random pipe pair (top and bottom pipes with random gap)"""
    # Random height for bottom pipe (determines gap position)
    size = random.randint(100, 300)
    # Create bottom pipe
    pipe = Pipe(False, xpos, size)
    # Create top pipe with consistent gap
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


# ==================== GAME INITIALIZATION ====================
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Load game images
BACKGROUND = pygame.image.load('game/assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load(
    'game/assets/sprites/message.png').convert_alpha()

# Create sprite groups for collision detection and rendering
bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

# Create ground sprites (two for continuous scrolling)
ground_group = pygame.sprite.Group()

for i in range(2):
    ground = Ground(GROUND_WIDHT * i)
    ground_group.add(ground)

# Create initial pipe pairs
pipe_group = pygame.sprite.Group()
for i in range(2):
    pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

# Game state variables
clock = pygame.time.Clock()

score = 0  # Player score (increments when passing pipes)
font = pygame.font.Font(None, 74)  # Font for score display

begin = True  # Flag to control menu/start screen loop

# ==================== MENU/START SCREEN LOOP ====================
while begin:

    clock.tick(60)

    # feed nn data
    # Get next pipe
    next_pipe = None
    for pipe in pipe_group:
        if pipe.rect[0] > bird.rect[0]:
            next_pipe = pipe
            break

    data = {
        'bird_x': bird.rect[0],
        'bird_y': bird.rect[1],
        'pipe_distance_x': next_pipe.rect[0] - bird.rect[0] if next_pipe else 0,
        'pipe_distance_y': next_pipe.rect[1] - bird.rect[1] if next_pipe else 0
    }

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()
                begin = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])

        new_ground = Ground(GROUND_WIDHT - 20)
        ground_group.add(new_ground)

    bird.begin()
    ground_group.update()

    bird_group.draw(screen)
    ground_group.draw(screen)

    pygame.display.update()


# ==================== MAIN GAME LOOP ====================
while True:

    clock.tick(fps)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()

    screen.blit(BACKGROUND, (0, 0))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])

        new_ground = Ground(GROUND_WIDHT - 20)
        ground_group.add(new_ground)

    if is_off_screen(pipe_group.sprites()[0]):
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])

        pipes = get_random_pipes(SCREEN_WIDHT * 2)

        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

        score += 1

    bird_group.update()
    ground_group.update()
    pipe_group.update()

    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    # Display score
    score_text = font.render(str(score), True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_WIDHT / 2 - 20, 50))

    pygame.display.update()

    if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
        pygame.mixer.music.load(hit)
        pygame.mixer.music.play()
        time.sleep(1)

        # ==================== GAME OVER SCREEN ====================
        # Display game over screen and wait for restart
        game_over = True
        while game_over:
            clock.tick(fps)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_SPACE or event.key == K_UP:
                        game_over = False

            screen.blit(BACKGROUND, (0, 0))
            screen.blit(BEGIN_IMAGE, (120, 150))
            bird_group.draw(screen)
            pipe_group.draw(screen)
            ground_group.draw(screen)

            pygame.display.update()

        # ==================== GAME RESET ====================
        # Reset score and all sprites for new game
        score = 0
        bird_group.empty()
        bird = Bird()
        bird_group.add(bird)

        pipe_group.empty()
        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        ground_group.empty()
        for i in range(2):
            ground = Ground(GROUND_WIDHT * i)
            ground_group.add(ground)

        continue
