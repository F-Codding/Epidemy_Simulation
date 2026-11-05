# -*- coding: utf-8 -*-
r"""Смоук: finalize создаёт sims\sim-<штамп> с CSV (с колонкой D) и PNG-графиком."""
import matplotlib
matplotlib.use("Agg")
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROG = str(REPO / "epid" / "epidemic_sim.py")
spec = importlib.util.spec_from_file_location("sim", PROG)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

cfg = dict(m.DEFAULTS)
cfg.update(mode=1, seed=42, gif=False, days=8)
sim = m.Epidemic(cfg)
viz = m.Viz(sim)
rec = m.Recorder(cfg)

# ход: полдня шагов, затем finalize
sim.advance(4)
sim.history.append(sim.bincount())
rec.finalize(sim)

d = rec.sim_dir
assert d.parent == (REPO / "epid" / "sims").resolve(), d
assert d.name.startswith("sim-"), d
csv = rec.csv_path
assert csv.exists(), csv
plot = rec.plot_path
assert plot.exists(), plot
lines = csv.read_text(encoding="utf-8").strip().splitlines()
assert lines[0] == "день,S,E,I,R,D", lines[0]
assert len(lines[1].split(",")) == 6, lines[1]   # 6 колонок, не 5
print("папка прогона:", d)
print("строка CSV:", lines[1])
print("PNG:", plot.stat().st_size, "байт")
print("СМОУК-ВЫВОД OK")
