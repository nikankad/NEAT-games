"""Generate charts for a training run.

Usage:
    python plot_run.py <run_id>

Charts are saved to:
    outputs/tron/<run_id>/fitness.png
    outputs/tron/<run_id>/kills_and_wins.png
    outputs/tron/<run_id>/diversity.png
"""

import sys
import os
import csv
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: python plot_run.py <run_id>')
    sys.exit(1)

run_id   = sys.argv[1]
base_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(base_dir, 'outputs', 'logs', f'training_{run_id}.csv')
out_dir  = os.path.join(base_dir, 'outputs', 'tron', run_id)

if not os.path.exists(log_file):
    print(f'Log file not found: {log_file}')
    sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

# ── Load CSV ──────────────────────────────────────────────────────────────────
with open(log_file) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print('Log file is empty.')
    sys.exit(1)

gens      = [int(r['gen'])                  for r in rows]
best_fit  = [float(r['best_fit'])           for r in rows]
avg_fit   = [float(r['avg_fit'])            for r in rows]
min_fit   = [float(r['min_fit'])            for r in rows]
avg_kills = [float(r['avg_kills'])          for r in rows]
bot_kills = [float(r['best_kills_vs_bots']) for r in rows]
n_species = [int(r['n_species'])            for r in rows]
hof_size  = [int(r['hof_size'])             for r in rows]

plt.style.use('dark_background')

# ── Chart 1 — Fitness ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gens, best_fit, label='best',  linewidth=2)
ax.plot(gens, avg_fit,  label='avg',   linewidth=2)
ax.plot(gens, min_fit,  label='min',   linewidth=1, linestyle='--', alpha=0.6)
ax.set(title=f'Fitness  |  run {run_id}', xlabel='Generation', ylabel='Fitness')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fitness.png'), dpi=150)
plt.close()

# ── Chart 2 — Kills ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.bar(gens, avg_kills, width=0.8, color='#e05050')
ax1.set(title='Avg kills / game', ylabel='Kills')
ax1.grid(alpha=0.2, axis='y')

ax2.plot(gens, bot_kills, color='#e09030', linewidth=2)
ax2.set(title='Best genome kills vs bots/HoF', xlabel='Generation', ylabel='Kills')
ax2.grid(alpha=0.2)

fig.suptitle(f'Kills  |  run {run_id}')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'kills_and_wins.png'), dpi=150)
plt.close()

# ── Chart 3 — Diversity ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(gens, n_species, color='#50a0e0', linewidth=2)
ax1.set(title='Species count', ylabel='Species')
ax1.grid(alpha=0.2)

ax2.plot(gens, hof_size, color='#a050e0', linewidth=2)
ax2.set(title='Hall of Fame size', xlabel='Generation', ylabel='Size')
ax2.grid(alpha=0.2)

fig.suptitle(f'Diversity  |  run {run_id}')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'diversity.png'), dpi=150)
plt.close()

print(f'Charts saved to {out_dir}/')
