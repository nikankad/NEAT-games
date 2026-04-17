"""Tron AI - NEAT-trained light cycles competing against each other.

Change NUM_PLAYERS at the top of the file to run with more or fewer AIs.
Supports 2–6 players.  Neural-network inputs stay the same regardless of
player count — each AI sees its own vision rays plus info about the nearest
alive opponent.
"""

import os
import sys
import numpy as np
import neat
import pygame
from pygame.locals import QUIT, KEYDOWN, K_SPACE, K_ESCAPE, K_s, K_1, K_2, K_3, K_4

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import utils.visualize as visualize

# ── Tune this to add more players ───────────────────────────────────────────────
NUM_PLAYERS = 4 # how many light cycles compete per game (2–6)

# ── Grid / display constants (scale with player count) ──────────────────────────
# More players → bigger grid so the arena doesn't feel cramped.
# Cell size shrinks to keep the window close to 600×600.
#   2 players → 40×40,  cell=15,  600px
#   3 players → 50×50,  cell=12,  600px
#   4 players → 60×60,  cell=10,  600px
#   5 players → 70×70,  cell= 8,  560px
#   6 players → 80×80,  cell= 7,  560px
GRID_W    = 20 + NUM_PLAYERS * 10
GRID_H    = GRID_W
CELL      = 600 // GRID_W
DISPLAY_W = CELL * GRID_W
DISPLAY_H = CELL * GRID_H
INFO_H      = 80
NET_PANEL_W = 260
SCREEN_W    = DISPLAY_W + NET_PANEL_W
SCREEN_H    = DISPLAY_H + INFO_H
MAX_STEPS = GRID_W * GRID_H // 2   # scales: more cells → longer games

# ── Colours ─────────────────────────────────────────────────────────────────────
BLACK   = (0,   0,   0)
BG      = (30,  30,  30)
GRIDCLR = (30,  30,  30)
BORDER  = (80,  80,  80)
TXT     = (180, 180, 180)

# Per-player colours — flat, distinct palette regardless of player count.
def _hsv_to_rgb(h, s, v):
    """h in [0,1), s/v in [0,1] → (r, g, b) ints 0-255."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def _make_palettes(n):
    colors, dead_colors, heads, labels = [], [], [], []
    for i in range(n):
        h = i / n
        colors.append(_hsv_to_rgb(h, 0.75, 0.85))       # flat trail
        dead_colors.append(_hsv_to_rgb(h, 0.30, 0.45))  # greyed-out dead trail
        heads.append((255, 255, 255))                    # white head, distinct from trail
        labels.append(f'P{i+1}')
    return colors, dead_colors, heads, labels

PLAYER_COLORS, PLAYER_DEAD_COLORS, PLAYER_HEADS, PLAYER_LABELS = _make_palettes(max(NUM_PLAYERS, 6))

# ── Directions: 0=RIGHT  1=DOWN  2=LEFT  3=UP ───────────────────────────────────
DV = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# ── Globals ──────────────────────────────────────────────────────────────────────
generation   = 0
best_genome  = None
hall_of_fame = []   # rolling list of past champion genomes
HOF_SIZE     = 5    # how many past champions to keep
SIMULATE     = '--simulate' in sys.argv   # headless fast mode
_scr = _clk = _fnt = None
_speed_level = 0   # 0=1x, 1=4x, 2=16x, 3=MAX
_SPEEDS      = [15, 60, None, None]   # tick fps per level (None = unlimited)
_DRAW_EVERY  = [1,  1,  4,    999999] # draw every Nth frame per level

# Leaderboard history — list of per-gen dicts for the terminal display
_leaderboard: list[dict] = []
_LEADERBOARD_ROWS = 15   # how many generations to show at once


# ── Terminal leaderboard ─────────────────────────────────────────────────────────

def _print_leaderboard(row: dict) -> None:
    """Append row to history and redraw the leaderboard in place."""
    _leaderboard.append(row)
    visible = _leaderboard[-_LEADERBOARD_ROWS:]

    # Move cursor up to overwrite previous board (skip on first print)
    if len(_leaderboard) > 1:
        lines_to_clear = _LEADERBOARD_ROWS + 4   # rows + header + separator + padding
        sys.stdout.write(f'\x1b[{lines_to_clear}A\x1b[J')

    # Header
    print('┌─────┬──────────┬──────────┬──────────────┬────────────────┬──────────┐')
    print('│ Gen │ Best fit │ Avg fit  │ Avg kills/gm │ Best vs bots   │ HoF wins │')
    print('├─────┼──────────┼──────────┼──────────────┼────────────────┼──────────┤')

    for r in visible:
        hof_str  = r['hof_win_rate'].ljust(8)
        bot_str  = f"{r['bot_kills']} kills".ljust(14)
        print(
            f"│{r['gen']:>4} "
            f"│{r['best_fit']:>9.0f} "
            f"│{r['avg_fit']:>9.0f} "
            f"│{r['avg_kills']:>13.2f} "
            f"│ {bot_str}"
            f"│ {hof_str}│"
        )

    # Pad empty rows so the table height stays fixed
    for _ in range(_LEADERBOARD_ROWS - len(visible)):
        print('│     │          │          │              │                │          │')

    print('└─────┴──────────┴──────────┴──────────────┴────────────────┴──────────┘')
    sys.stdout.flush()


# ── Display helpers ──────────────────────────────────────────────────────────────

_tiny_fnt = None   # 13px font for compact network labels

def _init_display():
    global _scr, _clk, _fnt, _tiny_fnt
    if _scr is not None:
        return
    pygame.init()
    _scr      = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('Tron AI')
    _clk      = pygame.time.Clock()
    _fnt      = pygame.font.Font(None, 26)
    _tiny_fnt = pygame.font.Font(None, 13)

_IN_LBL  = ['freeAhd','freeLft','freeRt','freeBck','oppDist','oppDx','oppDy','oppFree']
_OUT_LBL = ['L','Str','R']


def _act_col(v):
    """tanh value → flat colour.  green=positive, red=negative."""
    v = max(-1.0, min(1.0, float(v)))
    return (160, 60, 60) if v < 0 else (60, 160, 60)


def _net_layers(net):
    """Return {node: layer_index} with inputs at 0, outputs at max."""
    layer = {k: 0 for k in net.input_nodes}
    for node, _, _, _, _, links in net.node_evals:
        layer[node] = max((layer.get(s, 0) for s, _ in links), default=0) + 1
    # Ensure output nodes are always present and at the highest layer
    max_lv = max(layer.values(), default=0)
    for node in net.output_nodes:
        if node not in layer:
            layer[node] = max_lv + 1
        else:
            layer[node] = max(layer[node], max_lv)
    return layer


def _draw_one_net(net, alive, color_idx, px, py, pw, ph):
    """Draw a single compact network inside the rectangle (px,py,pw,ph)."""
    # ── Layout constants ──────────────────────────────────────────────────────
    LBL_L  = 28   # px reserved left  for input  labels
    LBL_R  = 16   # px reserved right for output labels
    TOP    = 16   # px reserved top   for player label
    BOT    = 4    # px reserved bottom
    R      = 4    # node radius

    layer  = _net_layers(net)

    # Group nodes by layer, inputs sorted by their index order
    by_lv  = {}
    for node, lv in layer.items():
        by_lv.setdefault(lv, []).append(node)
    # Keep input order stable; sort hidden/output by id
    input_order = {n: i for i, n in enumerate(net.input_nodes)}
    for lv in by_lv:
        by_lv[lv].sort(key=lambda n: input_order.get(n, n))

    # Pixel bounds for node area
    x0 = px + LBL_L
    x1 = px + pw - LBL_R - R
    y0 = py + TOP
    y1 = py + ph - BOT
    uw = max(x1 - x0, 1)
    uh = max(y1 - y0, 1)

    # Compute node pixel positions — spread layers evenly across width
    all_levels = sorted(by_lv.keys())
    n_levels   = max(len(all_levels) - 1, 1)   # avoid div-by-zero when only 1 layer
    pos = {}
    for rank, lv in enumerate(all_levels):
        nodes = by_lv[lv]
        nx    = x0 + int(rank / n_levels * uw)
        count = len(nodes)
        for idx, node in enumerate(nodes):
            ny = y0 + int((idx + 0.5) / count * uh)
            pos[node] = (nx, ny)

    # ── Draw connections ──────────────────────────────────────────────────────
    for node, _, _, _, _, links in net.node_evals:
        if node not in pos:
            continue
        for src, w in links:
            if src not in pos:
                continue
            col = (60, 120, 60) if w >= 0 else (120, 60, 60)
            pygame.draw.line(_scr, col, pos[src], pos[node], 1)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    values = getattr(net, 'values', {})
    for node, (nx, ny) in pos.items():
        val = values.get(node, 0.0)
        pygame.draw.circle(_scr, _act_col(val), (nx, ny), R)
        pygame.draw.circle(_scr, (50, 50, 50), (nx, ny), R, 1)

    # ── Input labels (left side) ──────────────────────────────────────────────
    for i, node in enumerate(net.input_nodes):
        if node not in pos:
            continue
        name = _IN_LBL[i] if i < len(_IN_LBL) else f'i{i}'
        surf = _tiny_fnt.render(name, True, (120, 120, 120))
        _scr.blit(surf, (px + 1, pos[node][1] - surf.get_height() // 2))

    # ── Output labels (right side) ────────────────────────────────────────────
    for i, node in enumerate(net.output_nodes):
        if node not in pos:
            continue
        name = _OUT_LBL[i] if i < len(_OUT_LBL) else f'o{i}'
        surf = _tiny_fnt.render(name, True, (120, 120, 120))
        _scr.blit(surf, (px + pw - LBL_R, pos[node][1] - surf.get_height() // 2))

    # ── Player label ──────────────────────────────────────────────────────────
    col  = PLAYER_COLORS[color_idx] if alive else PLAYER_DEAD_COLORS[color_idx]
    surf = _tiny_fnt.render(f'P{color_idx + 1}', True, col)
    _scr.blit(surf, (px + 2, py + 2))


def _draw_all_nets(nets, players, px, py, pw, ph):
    """Draw all player networks stacked in the right panel."""
    pygame.draw.rect(_scr, (22, 22, 22), (px, py, pw, ph))
    pygame.draw.line(_scr, BORDER, (px, py), (px, py + ph), 1)

    visible = [(net, p) for net, p in zip(nets, players) if net is not None]
    if not visible:
        return

    cell_h = ph // len(visible)
    for j, (net, p) in enumerate(visible):
        sub_y = py + j * cell_h
        # Divider between networks
        if j > 0:
            pygame.draw.line(_scr, (50, 50, 50),
                             (px, sub_y), (px + pw, sub_y), 1)
        _draw_one_net(net, p['alive'], p['color_idx'], px, sub_y, pw, cell_h)


def _draw(grid, dead_grid, players, step, nets=None):
    """Render one frame.  `players` is a list of dicts with x, y, alive, color_idx."""
    _scr.fill(BLACK)
    pygame.draw.rect(_scr, BG, (0, 0, DISPLAY_W, DISPLAY_H))

    # Dead trails first (dim, passable) then alive trails on top (bright, solid)
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            v = dead_grid[gy][gx]
            if v != 0:
                pygame.draw.rect(_scr, PLAYER_DEAD_COLORS[v - 1],
                                 (gx * CELL + 1, gy * CELL + 1, CELL - 2, CELL - 2))
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            v = grid[gy][gx]
            if v != 0:
                pygame.draw.rect(_scr, PLAYER_COLORS[v - 1],
                                 (gx * CELL + 1, gy * CELL + 1, CELL - 2, CELL - 2))

    # Heads
    for p in players:
        if p['alive']:
            pygame.draw.rect(_scr, PLAYER_HEADS[p['color_idx']],
                             (p['x'] * CELL, p['y'] * CELL, CELL, CELL))

    pygame.draw.rect(_scr, BORDER, (0, 0, DISPLAY_W, DISPLAY_H), 2)

    # Info panel (constrained to game area width)
    iy    = DISPLAY_H + 6
    col_w = DISPLAY_W // max(len(players), 1)
    _scr.blit(_fnt.render(f'Gen: {generation}   Step: {step}', True, TXT), (10, iy))
    for i, p in enumerate(players):
        label  = PLAYER_LABELS[p['color_idx']]
        status = 'ALIVE' if p['alive'] else 'DEAD'
        surf   = _fnt.render(f'{label}: {status}', True, PLAYER_COLORS[p['color_idx']])
        _scr.blit(surf, (10 + i * col_w, iy + 28))
    speed_label = ['1x', '4x', '16x', 'MAX'][_speed_level]
    _scr.blit(_fnt.render(f'SPACE: speed [{speed_label}]  ESC: quit', True, (70, 70, 100)), (10, iy + 54))

    # Network panel — all playing genomes stacked
    if nets is not None:
        _draw_all_nets(nets, players, DISPLAY_W, 0, NET_PANEL_W, SCREEN_H)

    pygame.display.flip()


# ── Game logic ───────────────────────────────────────────────────────────────────


def _ray_steps(grid, x, y, dx, dy):
    """Steps to nearest wall/trail in direction (dx,dy)."""
    cx, cy = x + dx, y + dy
    dist   = 0
    while 0 <= cx < GRID_W and 0 <= cy < GRID_H and grid[cy][cx] == 0:
        dist += 1
        cx   += dx
        cy   += dy
    return dist


def _get_inputs(me, players, grid):
    """
    8-element input vector — wall distances + opponent info.

    [0]  dist ahead      — free steps ahead  (0.0=wall right there, 1.0=fully open)
    [1]  dist left       — free steps left
    [2]  dist right      — free steps right
    [3]  dist behind     — free steps behind
    [4]  opp dist        — normalised Manhattan distance to nearest opponent (0=on top, 1=far)
    [5]  opp dx          — (opp.x - me.x) / GRID_W  — signed relative position
    [6]  opp dy          — (opp.y - me.y) / GRID_H
    [7]  opp free ahead  — how much free space the opponent has in front of them (0=trapped, 1=open)
    """
    x, y  = me['x'], me['y']
    md    = me['dir']
    MAX_D = max(GRID_W, GRID_H)

    ahead_dx, ahead_dy = DV[md]
    left_dx,  left_dy  = DV[(md + 3) % 4]
    right_dx, right_dy = DV[(md + 1) % 4]
    back_dx,  back_dy  = DV[(md + 2) % 4]

    dist_ahead  = _ray_steps(grid, x, y, ahead_dx, ahead_dy) / MAX_D
    dist_left   = _ray_steps(grid, x, y, left_dx,  left_dy)  / MAX_D
    dist_right  = _ray_steps(grid, x, y, right_dx, right_dy) / MAX_D
    dist_behind = _ray_steps(grid, x, y, back_dx,  back_dy)  / MAX_D

    alive_opps = [p for p in players if p is not me and p['alive']]
    if alive_opps:
        nearest      = min(alive_opps, key=lambda p: abs(p['x'] - x) + abs(p['y'] - y))
        raw_dist     = abs(nearest['x'] - x) + abs(nearest['y'] - y)
        opp_dist     = raw_dist / (GRID_W + GRID_H - 2)
        opp_dx       = (nearest['x'] - x) / GRID_W
        opp_dy       = (nearest['y'] - y) / GRID_H
        odx, ody     = DV[nearest['dir']]
        opp_free     = _ray_steps(grid, nearest['x'], nearest['y'], odx, ody) / MAX_D
    else:
        opp_dist = 1.0; opp_dx = 0.0; opp_dy = 0.0; opp_free = 1.0

    return [dist_ahead, dist_left, dist_right, dist_behind,
            opp_dist, opp_dx, opp_dy, opp_free]   # length 8


def _scripted_action(p, grid):
    """Wall-avoider bot: straight → right → left → give up."""
    for offset in [0, 1, 3, 2]:
        d      = (p['dir'] + offset) % 4
        dx, dy = DV[d]
        nx, ny = p['x'] + dx, p['y'] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and grid[ny][nx] == 0:
            if offset == 0: return 1
            if offset == 1: return 2
            if offset == 3: return 0
    return 1


def _starting_positions(n):
    """
    Place players in opposite quadrants facing the center.
    Facing inward means going straight keeps them alive longest — the network
    has a clear gradient: turn before hitting the far wall = more steps = higher fitness.
    """
    import random
    # Spawn players near opposite edges, facing along the wall (parallel to the border).
    # Facing along the wall means:
    #   - going straight quickly hits a corner → death in ~GRID_W steps
    #   - turning inward opens up the whole grid → much longer survival
    #   - players start far apart so they don't magnetically converge early
    edge = 3

    # Each slot: (x, y, direction)
    # Directions: 0=RIGHT 1=DOWN 2=LEFT 3=UP
    slots = [
        (edge,              edge,              0),   # top-left,     facing right (along top wall)
        (GRID_W - 1 - edge, GRID_H - 1 - edge, 2),  # bottom-right, facing left  (along bottom wall)
        (edge,              GRID_H - 1 - edge, 0),  # bottom-left,  facing right (along bottom wall)
        (GRID_W - 1 - edge, edge,              2),  # top-right,    facing left  (along top wall)
    ]
    random.shuffle(slots)

    positions = []
    for i in range(n):
        sx, sy, d = slots[i % len(slots)]
        sx = max(2, min(GRID_W - 3, sx + random.randint(-2, 2)))
        sy = max(2, min(GRID_H - 3, sy + random.randint(-2, 2)))
        positions.append((sx, sy, d))
    return positions


def run_game(nets, render=False, return_stats=False):
    """
    Run one Tron game with len(nets) players.
    Pass None in the list to use the scripted-bot for that slot.
    Returns a list of fitness scores, one per player.
    If return_stats=True, returns (scores, kills) instead.
    """
    if render:
        _init_display()

    n         = len(nets)
    grid      = [[0] * GRID_W for _ in range(GRID_H)]   # live trails (blocks movement)
    dead_grid = [[0] * GRID_W for _ in range(GRID_H)]   # dead trails (visual only)

    starts  = _starting_positions(n)
    players = [
        {'x': x, 'y': y, 'dir': d, 'alive': True, 'color_idx': i,
         'trail': [(x, y)]}
        for i, (x, y, d) in enumerate(starts)
    ]

    for p in players:
        grid[p['y']][p['x']] = p['color_idx'] + 1

    step         = 0
    kill_steps   = [[] for _ in range(n)]   # step number of each kill per player
    death_step   = [-1] * n                 # step when player died (-1 = alive/survived)
    wall_death   = [False] * n              # True if killed by border or own trail
    last_turned  = [-2] * n                 # step of last turn (-2 so turn is free on step 0)
    close_steps  = [0] * n                  # steps spent within CLOSE_DIST of an opponent

    for step in range(MAX_STEPS):
        if render:
            global _speed_level
            for e in pygame.event.get():
                if e.type == QUIT:
                    pygame.quit(); sys.exit()
                if e.type == KEYDOWN:
                    if e.key in (K_SPACE, K_s):
                        _speed_level = (_speed_level + 1) % len(_SPEEDS)
                    if e.key == K_ESCAPE: pygame.quit(); sys.exit()
            fps = _SPEEDS[_speed_level]
            if fps is not None:
                _clk.tick(fps)

        alive_count = sum(1 for p in players if p['alive'])
        if alive_count <= 1:
            break

        # ── Track proximity — penalise circling near opponent without killing ────
        CLOSE_DIST = 6
        for i, p in enumerate(players):
            if not p['alive']:
                continue
            for other in players:
                if other is p or not other['alive']:
                    continue
                if abs(p['x'] - other['x']) + abs(p['y'] - other['y']) <= CLOSE_DIST:
                    close_steps[i] += 1
                    break

        # ── Decide actions ───────────────────────────────────────────────────────
        actions = []
        for i, p in enumerate(players):
            if not p['alive']:
                actions.append(None)
                continue
            net = nets[i]
            if net is not None:
                inp    = _get_inputs(p, players, grid)
                out    = net.activate(inp)
                action = int(np.argmax(out))
            else:
                action = _scripted_action(p, grid)
            actions.append(action)

        # ── Apply turns (1-step cooldown prevents same-step double-turns) ─────────
        TURN_COOLDOWN = 1
        for i, p in enumerate(players):
            if not p['alive'] or actions[i] is None:
                continue
            a = actions[i]
            if a == 1:   # straight — always allowed
                continue
            if step - last_turned[i] < TURN_COOLDOWN:
                continue
            if   a == 0: p['dir'] = (p['dir'] + 3) % 4   # left
            elif a == 2: p['dir'] = (p['dir'] + 1) % 4   # right
            last_turned[i] = step

        # ── Move ─────────────────────────────────────────────────────────────────
        new_pos = {}
        for i, p in enumerate(players):
            if not p['alive']:
                continue
            dx, dy = DV[p['dir']]
            nx, ny = p['x'] + dx, p['y'] + dy
            if (nx, ny) in new_pos:
                new_pos[(nx, ny)].append(i)   # head-on collision
            else:
                new_pos[(nx, ny)] = [i]

        # ── Collisions ───────────────────────────────────────────────────────────
        for (nx, ny), idxs in new_pos.items():
            oob       = not (0 <= nx < GRID_W and 0 <= ny < GRID_H)
            head_on   = len(idxs) > 1
            trail_own = None if oob else (grid[ny][nx] - 1 if grid[ny][nx] != 0 else None)
            hits      = oob or trail_own is not None or head_on
            for i in idxs:
                if hits:
                    players[i]['alive'] = False
                    death_step[i] = step
                    if oob or trail_own == i:   # border or self-trail = wall death
                        wall_death[i] = True
                    # Credit trail kill: ran into someone else's trail
                    if trail_own is not None and trail_own != i and not head_on:
                        kill_steps[trail_own].append(step)
                    # Credit head-on kill: whoever was directly facing the opponent wins
                    if head_on and len(idxs) == 2:
                        other    = idxs[1] if idxs[0] == i else idxs[0]
                        ddx, ddy = DV[players[i]['dir']]
                        facing   = (players[i]['x'] + ddx == players[other]['x'] and
                                    players[i]['y'] + ddy == players[other]['y'])
                        if facing:
                            kill_steps[i].append(step)
                    # Move dead trail to dead_grid so living players can pass through
                    val = players[i]['color_idx'] + 1
                    for tx, ty in players[i]['trail']:
                        if grid[ty][tx] == val:
                            dead_grid[ty][tx] = val
                            grid[ty][tx]      = 0
                else:
                    players[i]['x'] = nx
                    players[i]['y'] = ny
                    players[i]['trail'].append((nx, ny))
                    grid[ny][nx] = players[i]['color_idx'] + 1

        if render and step % _DRAW_EVERY[_speed_level] == 0:
            _draw(grid, dead_grid, players, step, nets)

        if sum(1 for p in players if p['alive']) <= 1:
            break

    # ── Fitness ──────────────────────────────────────────────────────────────────
    # +1 per frame survived — continuous gradient, wall deaths cost all future frames
    # +20 for winning — last player alive gets a flat bonus
    # Nothing else.
    WIN_BONUS = 20

    kill_counts = [len(ks) for ks in kill_steps]
    scores = []
    for i in range(n):
        alive_steps = death_step[i] if death_step[i] >= 0 else step
        winner = (death_step[i] == -1) or (
            death_step[i] >= 0 and
            all(death_step[j] >= 0 and death_step[j] < death_step[i]
                for j in range(n) if j != i)
        )
        scores.append(alive_steps + (WIN_BONUS if winner else 0))

    if return_stats:
        return scores, kill_counts
    return scores


# ── NEAT evaluation ──────────────────────────────────────────────────────────────

def eval_genomes(genomes, config):
    global generation, best_genome, hall_of_fame
    generation += 1

    genome_list = [g for _, g in genomes]
    net_list    = [neat.nn.FeedForwardNetwork.create(g, config) for g in genome_list]
    n_genomes   = len(genome_list)

    for g in genome_list:
        g.fitness = 0.0

    hof_nets = [neat.nn.FeedForwardNetwork.create(g, config) for g in hall_of_fame]

    # ── Round 1: peer games ───────────────────────────────────────────────────────
    # Each genome plays 2 shuffled peer games to reduce matchup variance.
    # Scores are accumulated and averaged at the end.
    import random
    peer_scores = [0.0] * n_genomes
    all_kills   = [0]   * n_genomes
    peer_counts = [0]   * n_genomes   # how many peer games each genome played

    first_match = not SIMULATE
    for _ in range(3):
        order = list(range(n_genomes))
        random.shuffle(order)
        for i in range(0, n_genomes, NUM_PLAYERS):
            group_idx  = order[i : i + NUM_PLAYERS]
            group_nets = [net_list[j] for j in group_idx]
            # Pad short last group with scripted bots
            while len(group_nets) < NUM_PLAYERS:
                group_nets.append(None)
            scores, kills = run_game(group_nets, render=first_match, return_stats=True)
            first_match   = False
            for slot, j in enumerate(group_idx):
                peer_scores[j] += scores[slot]
                all_kills[j]   += kills[slot]
                peer_counts[j] += 1

    # Average over the games played
    for j in range(n_genomes):
        if peer_counts[j] > 0:
            peer_scores[j] /= peer_counts[j]

    # ── Round 2: Hall of Fame games ───────────────────────────────────────────────
    # Every genome plays against the fixed HoF — stable absolute signal.
    hof_scores = [0.0] * n_genomes
    if hof_nets:
        for idx, net in enumerate(net_list):
            slots_needed = NUM_PLAYERS - 1
            opponents: list[object] = list(hof_nets[:slots_needed])
            while len(opponents) < slots_needed:
                opponents.append(None)
            scores, _ = run_game([net] + opponents, render=False, return_stats=True)
            hof_scores[idx] = scores[0]

    # Combine: peer games (3 rounds averaged) are now the primary signal
    for idx, g in enumerate(genome_list):
        if hof_nets:
            g.fitness = peer_scores[idx] * 0.7 + hof_scores[idx] * 0.3
        else:
            g.fitness = peer_scores[idx]

    # ── Update Hall of Fame — add the genome with the most kills this gen ────────
    # Selecting by kills (not fitness) ensures HoF opponents are actual fighters,
    # so training against them rewards combat rather than passive survival.
    best_genome    = max(genome_list, key=lambda g: g.fitness)
    killiest_idx   = max(range(n_genomes), key=lambda i: all_kills[i])
    killiest_genome = genome_list[killiest_idx]
    hof_candidate  = killiest_genome if all_kills[killiest_idx] > 0 else best_genome
    hall_of_fame.append(hof_candidate)
    if len(hall_of_fame) > HOF_SIZE:
        hall_of_fame.pop(0)

    # ── Baseline: best genome vs previous HoF (or bots if HoF empty) ────────────
    best_net = net_list[genome_list.index(best_genome)]
    if hof_nets:
        baseline_opponents: list = list(hof_nets[:NUM_PLAYERS - 1])
        while len(baseline_opponents) < NUM_PLAYERS - 1:
            baseline_opponents.append(None)
    else:
        baseline_opponents = [None] * (NUM_PLAYERS - 1)
    _, baseline_kills = run_game([best_net] + baseline_opponents, render=False, return_stats=True)
    bot_kill_count    = baseline_kills[0]

    # ── HoF win rate ──────────────────────────────────────────────────────────────
    hof_wins = 0
    for hof_net in hof_nets:
        hof_opponents = [hof_net] + [None] * (NUM_PLAYERS - 2)
        scores, _ = run_game([best_net] + hof_opponents, render=False, return_stats=True)
        if scores[0] > scores[1]:
            hof_wins += 1

    # ── Print diagnostics ─────────────────────────────────────────────────────────
    best_fit     = max(g.fitness for g in genome_list)
    avg_fit      = sum(g.fitness for g in genome_list) / n_genomes
    avg_kills    = sum(all_kills) / n_genomes
    hof_win_rate = f"{hof_wins}/{len(hof_nets)}" if hof_nets else "n/a"

    if SIMULATE:
        _print_leaderboard({
            'gen':          generation,
            'best_fit':     best_fit,
            'avg_fit':      avg_fit,
            'avg_kills':    avg_kills,
            'bot_kills':    bot_kill_count,
            'hof_win_rate': hof_win_rate,
        })
    else:
        print(
            f"Gen {generation:>4} | "
            f"fit best={best_fit:.0f} avg={avg_fit:.0f} | "
            f"avg kills/game={avg_kills:.2f} | "
            f"best vs bots: {bot_kill_count} kills | "
            f"hof wins={hof_win_rate} | "
            f"hof size={len(hall_of_fame)}"
        )


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'config-tron.txt')
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(config)
    if not SIMULATE:
        p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 200)
    print(f'\nBest genome – fitness: {winner.fitness:.2f}')

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    node_names = {
        -1: 'ahead', -2: 'left',  -3: 'right',   -4: 'behind',
        -5: 'oppDist', -6: 'oppDx', -7: 'oppDy', -8: 'oppFree',
         0: 'Turn L', 1: 'Straight', 2: 'Turn R',
    }
    visualize.draw_net(config, winner, True,
                       filename=os.path.join(out_dir, 'winner_net'),
                       node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True,
                         filename=os.path.join(out_dir, 'fitness.svg'))
