import sys
sys.path.insert(0, ".")
from training.self_play import run_games
r = run_games(num_games=50, agent_a="heuristic", agent_b="random")
print(f"Done: {r['wins_a']}W / {r['wins_b']}L / {r['draws']}D  win_rate={r['win_rate_a']:.1%}")
print(f"Logs written: {len(r['log_ids'])}")
