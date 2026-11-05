# -*- coding: utf-8 -*-
"""Головные тесты: родное Tk-окно настроек, флаг dead_infect, потолок дней."""
import matplotlib
matplotlib.use("Agg")   # окно симуляции не участвует; диалог теперь на Tk
import py_compile
from pathlib import Path

PROG = str(Path(__file__).resolve().parents[1] / "epid" / "epidemic_sim.py")

py_compile.compile(PROG, doraise=True)
print("py_compile OK")

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("sim", PROG)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def set_ent(dialog, key, text):
    e = dialog["entries"][key]
    e.delete(0, "end")
    e.insert(0, text)


# --- 1) валидация и разбор полей ---
d = m._tk_start_dialog(show=False)
set_ent(d, "n", "abc")
d["on_start"]()
assert d["cfg"] is None and "не число" in d["err_lbl"].cget("text")
print("ошибка 'не число' показана: OK")

set_ent(d, "n", "100")
set_ent(d, "n_initial", "5000")
d["on_start"]()
assert d["cfg"] is None and "больше численности" in d["err_lbl"].cget("text")
print("перекрёстная проверка работает: OK")

# успех: огромный days (потолка нет), пустой seed -> None, режим, флаги
set_ent(d, "n", "100")
set_ent(d, "n_initial", "3")
set_ent(d, "days", "100000")
set_ent(d, "seed", "")
d["dead_var"] and d["dead_var"].set(True)   # флаг от мёртвых
d["mode_var"].set(2)                        # ручные шаги
d["on_start"]()
cfg = d["cfg"]
assert cfg is not None
print("cfg из окна:", {k: cfg[k] for k in ("n", "days", "seed", "dead_infect", "mode")})
assert cfg["days"] == 100000 and cfg["seed"] is None
assert cfg["dead_infect"] is True and cfg["mode"] == 2
try:
    d["root"].update()
except Exception:
    pass   # root уже уничтожен самим диалогом — так и задумано
print("диалог собирает cfg: OK")

# --- 2) флаг включён: мёртвый в плотной толпе заражает ---
cfg2 = dict(m.DEFAULTS)
cfg2.update(n=60, seed=7, days=10**9, gif=False, lethality=1.0, dead_infect=True)
s = m.Epidemic(cfg2)
s.pos[:] = 5.0
s.velocity[:] = 0.0
s.state[:] = 0
s.timer[:] = 0
s.state[0] = 4
s.history = [s.bincount()]
for k in range(10 * cfg2["steps_per_day"]):
    s.step()
assert (s.state == 1).sum() + (s.state == 2).sum() + (s.state == 3).sum() > 0
h = np.array(s.history)
assert h.shape[1] == 5
print("флаг ON: мёртвый заразил:", (s.state == 1).sum(), "/",
      (s.state == 2).sum(), "/", (s.state == 3).sum())

# --- 3) флаг выключен: мёртвые НЕ заражают (регресс) ---
cfg3 = dict(cfg2)
cfg3["dead_infect"] = False
s3 = m.Epidemic(cfg3)
s3.pos[:] = 5.0
s3.velocity[:] = 0.0
s3.state[:] = 0
s3.timer[:] = 0
s3.state[0] = 4
s3.history = [s3.bincount()]
for k in range(10 * cfg2["steps_per_day"]):
    s3.step()
assert (s3.state == 1).sum() == 0 and (s3.state == 2).sum() == 0 and (s3.state == 3).sum() == 0
print("флаг OFF: заражений от мёртвого нет: OK")

# --- 4) дни без потолка ---
cfg4 = dict(m.DEFAULTS)
cfg4.update(days=1_000_000, gif=False, seed=42)
s4 = m.Epidemic(cfg4)
assert s4.done is False
s4.advance(2)
assert s4.day == 2
print("days=1000000 принимается: OK")

print("ВСЕ ТЕСТЫ OK")
