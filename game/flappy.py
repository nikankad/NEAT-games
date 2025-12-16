# Flappy Bird Game - Main Game Loop
import pygame
import random
import time
import neat
import visualize

from pygame.locals import *
# from model import NeatModel
# ==================== GAME CONSTANTS ====================
# Screen dimensions
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600

# Bird movement constants
SPEED = 20  # Flap strength
GRAVITY = 2.0  # Falling acceleration
GAME_SPEED =40  # Speed of pipes and ground moving

# Ground dimensions
GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

# Pipe dimensions
PIPE_WIDHT = 80
PIPE_HEIGHT = 500

# Gap between upper and lower pipes
PIPE_GAP =  200

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
        bird_color = random.choice(['bluebird', 'redbird', 'yellowbird'])
        self.images = [pygame.image.load(f'game/assets/sprites/{bird_color}-upflap.png').convert_alpha(),
                   pygame.image.load(f'game/assets/sprites/{bird_color}-midflap.png').convert_alpha(),
                   pygame.image.load(f'game/assets/sprites/{bird_color}-downflap.png').convert_alpha()]

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
        self.inverted = inverted
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



def is_off_screen(sprite):
    """Check if a sprite has moved off the left side of the screen"""
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    """Generate a random pipe pair (top and bottom pipes with random gap)"""
    size = random.randint(150, 400)
    # Create bottom pipe
    pipe = Pipe(False, xpos, size)
    # Create top pipe with consistent gap
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted



def run_bird(genomes, config):
    """
    Runs the Flappy Bird game with AI-controlled birds using NEAT algorithm.
    This function initializes a Pygame environment and simulates multiple birds
    controlled by neural networks evolved via NEAT (NeuroEvolution of Augmenting
    Topologies). Each bird's behavior is determined by its neural network, which
    takes visual inputs and decides when to jump.
    Args:
        genomes (list): List of tuples containing (genome_id, genome_object) from NEAT.
        config (neat.config.Config): NEAT configuration object containing network parameters.
    Returns:
        None
    Game Mechanics:
        - Birds are controlled by neural networks that receive 4 inputs:
          1. Bird's Y position (vertical screen position)
          2. Horizontal distance to next pipe
          3. Vertical distance to next pipe
          4. Bird's current velocity (negative = falling, positive = rising)
        - Output > 0.5 triggers a jump (bump)
        - Fitness increases when birds successfully pass pipe pairs
        - Birds are removed when colliding with pipes or ground
    Note:
        - Consider updating fitness incrementally when pipes are passed (current approach)
          rather than only at the end, as this provides better feedback for evolution
        - The function has a global 'generation' variable reference
        - Currently has a bug: resetting only one bird during collision instead of
          removing that specific bird from the population
    """
    nets = []
    ge = []
    birds = []
  #init game
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
    # bird = Bird()
    # bird_group.add(bird)

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
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0 
        #init bird
        bird = Bird()
        birds.append(bird)
        bird_group.add(bird)
        ge.append(g)          # ← MISSING LINE

        #MAIN LOOP
    global generation


        
    score = 0  # Player score (increments when passing pipes)
    font = pygame.font.Font(None, 74)  # Font for score display
    # Game state variables
    clock = pygame.time.Clock()

    # ==================== MAIN GAME LOOP ====================
    while birds: 
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        #input data and get result from network 
        for index, bird in enumerate(birds):
            next_pipe = None
            for pipe in pipe_group:
                if not pipe.inverted and pipe.rect.right > bird.rect.left:
                    next_pipe = pipe
                    break
            # Neural network inputs:
            # [0] Bird's Y position (vertical position on screen)
            # [1] Horizontal distance to next pipe (gap between bird and pipe)
            # [2] Vertical distance to next pipe (how far above/below the pipe gap the bird is)
            # [3] Bird's current velocity (negative = falling, positive = rising)
            inputs = [
                bird.rect[1] / SCREEN_HEIGHT,                          # 0-1 range
                (next_pipe.rect[0] - bird.rect[0]) / SCREEN_WIDHT,    # 0-1 range
                (next_pipe.rect.top - PIPE_GAP / 2 - bird.rect.centery) / SCREEN_HEIGHT,    # normalized
                bird.speed / 20                                          # normalized
            ]
            output = nets[index].activate(inputs)
            if output[0] > 0.5:
                    bird.bump()

    
        # update sprites
        bird_group.update()
        pipe_group.update()
        ground_group.update()
        for i in range(len(ge)):
            ge[i].fitness += 0.05            
        screen.blit(BACKGROUND, (0, 0))

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])

            pipes = get_random_pipes(SCREEN_WIDHT * 2)

            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])
            for g in ge:
                g.fitness += 5
            score += 1
            # EARLY STOP CONDITION
            # if score >= 4:
            #     for g in ge:
            #         g.fitness+=10
            #     return
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])

            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)
            

        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)
        for i in reversed(range(len(birds))):
                    bird = birds[i]
                    hit_ground = pygame.sprite.spritecollideany(bird, ground_group, pygame.sprite.collide_mask)
                    hit_pipe = pygame.sprite.spritecollideany(bird, pipe_group, pygame.sprite.collide_mask)
                    out_of_bounds = (bird.rect[1] > SCREEN_HEIGHT or bird.rect[1] < 0)

                    if(hit_ground or hit_pipe or out_of_bounds):
                        ge[i].fitness -= 1
                        bird_group.remove(bird)
                        birds.pop(i)
                        nets.pop(i)
                        ge.pop(i)

        score_text = font.render(str(score), True, (255, 255, 255))
        screen.blit(score_text, (SCREEN_WIDHT / 2 - 20, 50))

        pygame.display.update()

if __name__ == "__main__":
    # Set configuration file
    config_path = "model\config-flappybird.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(run_bird, 200)
    print("Best fitness:", winner.fitness)
    visualize.draw_net(
        config,
        winner,
        view=True,
        filename="best_net.gv",
        node_names={
            0: "Bird Y Position",
            1: "Distance to Pipe",
            2: "Distance to Gap",
            3: "Bird Velocity",
            4: "Flap Decision"
        }
    )
    # visualize.plot_stats(stats, ylog=False, view=True, filename="fitness_plot.svg")

    visualize.plot_species(stats, view=True, filename="species_plot.svg")