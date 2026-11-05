# -*- coding: utf-8 -*-
"""Смоук: авто-панель (ход по времени), ручная панель и запись с новым конфигом."""
import matplotlib
matplotlib.use("Agg")
import importlib.util
import time
from pathlib import Path

PROG = str(Path(__file__).resolve().parents[1] / "epid" / "epidemic_sim.py")
spec = importlib.util.spec_from_file_location("sim", PROG)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

cfg = dict(m.DEFAULTS)
cfg.update(mode=1, seed=42, gif=False, dead_infect=True, days=40)
sim = m.Epidemic(cfg)
viz = m.Viz(sim)
rec = m.Recorder(cfg)
a = m._auto_controls(sim, viz, rec)

# старт: первый кадр синхронизирует часы и НЕ двигает симуляцию (день 0)
a["state"]["t"] = None
a["update"]()
assert sim.day == 0, f"старт прыгнул на день {sim.day}"
a["update"]()   # второй кадр сразу после первого — тоже не должно прыгнуть
assert sim.day == 0, f"долг после первого кадра: день {sim.day}"
print("старт ровно с дня 0: OK")

# скорость: dps=50 -> за 0.2с должно пройти ~10 дней
a["state"]["dps"] = 50.0
a["state"]["t"] = time.perf_counter() - 0.2     # имитируем 0.2 c простоя
a["update"]()
adv = sim.day
print("за 0.2 c пройдено дней:", adv)
assert 6 <= adv <= 14, adv

# пауза: ход замирает, аккумулятор не копится
a["state"]["paused"] = True
a["state"]["t"] = time.perf_counter() - 1.0
a["update"]()
assert sim.day == adv
a["state"]["paused"] = False
a["state"]["t"] = time.perf_counter()           # сброс: не догонять пропущенное
a["update"]()
print("пауза не копит пропущенных дней: OK")

# ручная панель
sim2 = m.Epidemic(cfg)
viz2 = m.Viz(sim2)
rec2 = m.Recorder(cfg)
b = m._manual_controls(sim2, viz2, rec2)
b["on_step"]()
assert sim2.day == 1, sim2.day
print("авто-панель (ход по времени) и ручная панель: OK")
print("СМОУК OK")
