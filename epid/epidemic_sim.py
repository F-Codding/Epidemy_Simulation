# -*- coding: utf-8 -*-
"""
Агентная симуляция эпидемии (SEIR) с настройкой параметров перед запуском,
двумя режимами работы (авто-анимация / ручные шаги) и записью GIF по шагам.

Состояния агентов:
  S (Susceptible) - здоровый, но восприимчивый      -> синий
  E (Exposed)     - инкубационный период, заразен   -> жёлтый
  I (Infected)    - болезнь, продолжает заражать    -> красный
  R (Removed)     - переболел и получил иммунитет   -> зелёный
  D (Dead)        - погиб, остаётся на месте        -> чёрный

Зависимости: numpy, matplotlib, pillow
Запуск:      python epidemic_sim.py   (все параметры задаются в стартовом окне)

Управление окном (режим 1 «авто») — кнопки прямо на панели + горячие клавиши:
  скорость      - задаётся в окне настроек (дней симуляции в секунду);
                  клавиши +/- удваивают/делят её на ходу
  Пауза/Пуск    - Space или p
  Кадр вперёд   - Right (работает и на паузе: покадровый просмотр)
  GIF: вкл/выкл - g
  Сохранить GIF - s
  Выход         - q / Esc

Управление окном (режим 2 «вручную») — кнопки на панели + клавиши:
  слайдер       - интервал шага, дней (лог-шкала)
  Шаг           - Right / Space: продвинуть симуляцию на интервал
  GIF / Сохранить / Выход - то же, что в авто-режиме

"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

# чтобы кириллица корректно печаталась в консоль Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# pillow нужен только для записи GIF; без него всё остальное работает
try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

# scipy ускоряет поиск контактов KD-деревом на больших популяциях (необязателен)
try:
    from scipy.spatial import cKDTree
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# порог (пар больной x здоровый), выше которого используется KD-дерево
KD_PAIRS_THRESHOLD = 1_000_000

HERE = Path(__file__).resolve().parent

# =================== ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ =====================
DEFAULTS = {
    "n": 500,             # численность населения
    "world": 10.0,        # размер мира (квадрат WORLD x WORLD)
    "days": 120,          # длительность симуляции, дней
    "speed": 0.40,        # средняя скорость перемещения агента (единиц/день)
    "radius": 0.35,       # радиус тесного контакта
    "infect_p": 0.25,     # вероятность заражения за день контакта в радиусе
    "incub_lo": 2,        # инкубационный период: мин дней
    "incub_hi": 5,        # инкубационный период: макс дней
    "ill_lo": 6,          # длительность болезни: мин дней
    "ill_hi": 14,         # длительность болезни: макс дней
    "immunity": 0.95,     # шанс получить иммунитет после болезни
    "lethality": 0.02,    # смертность болезни: доля погибших среди выздоравливающих
    "dead_infect": False, # флаг: мёртвые продолжают заражать в радиусе контакта
    "n_initial": 5,       # число "нулевых пациентов"
    "steps_per_day": 8,   # внутренних шагов симуляции в одном "дне"
    "seed": 42,           # seed случайности (Enter+пусто = None)
    "mode": 1,            # 1 = авто-анимация, 2 = ручные шаги
    "dps": 2.5,           # скорость автоанимации: дней симуляции в секунду
    "step_days": 1,       # интервал ручного шага, дней (дни/нажатие)
    "gif": True,          # записывать ли GIF
    "gif_every": 8,       # кадр GIF каждые N отображаемых кадров (в авто ~1 день)
    "gif_scale": 0.5,     # масштаб кадра GIF (0.5 = половина размера окна)
}
COLORS = ["tab:blue", "gold", "tab:red", "seagreen", "black"]
LABELS = ["Здоровые (S)", "Инкубация (E)", "Больные (I)", "Переболевшие (R)", "Умершие (D)"]
# ===============================================================


# ------------------------- стартовое окно настроек (GUI) -------------------------
# FIELDS: (подпись, ключ конфига, значение по умолчанию, cast, проверка, пояснение)
# Потолка по дням больше нет: для days достаточно >= 1.
FIELDS = [
    ("Численность населения",    "n",             500,  int,   lambda v: v >= 2,              "нужно >= 2"),
    ("Размер мира, сторона",     "world",         10.0, float, lambda v: v > 1,               "нужно > 1"),
    ("Длительность, дней",       "days",          120,  int,   lambda v: v >= 1,              "нужно >= 1"),
    ("Скорость агентов, ед/день","speed",         0.40, float, lambda v: v >= 0,              "нужно >= 0"),
    ("Радиус контакта",          "radius",        0.35, float, lambda v: v > 0,               "нужно > 0"),
    ("Вероятность заражения",    "infect_p",      0.25, float, lambda v: 0 < v <= 1,          "нужно 0..1"),
    ("Инкубация: мин, дней",     "incub_lo",      2,    int,   lambda v: v >= 0,              "нужно >= 0"),
    ("Инкубация: макс, дней",    "incub_hi",      5,    int,   lambda v: v >= 0,              "нужно >= 0"),
    ("Болезнь: мин, дней",       "ill_lo",        6,    int,   lambda v: v >= 1,              "нужно >= 1"),
    ("Болезнь: макс, дней",      "ill_hi",        14,   int,   lambda v: v >= 1,              "нужно >= 1"),
    ("Шанс иммунитета (0..1)",   "immunity",      0.95, float, lambda v: 0 <= v <= 1,         "нужно 0..1"),
    ("Смертельность (0..1)",     "lethality",     0.02, float, lambda v: 0 <= v < 1,          "нужно 0..0.999"),
    ("Нулевых пациентов",        "n_initial",     5,    int,   lambda v: v >= 1,              "нужно >= 1"),
    ("Внутр. шагов в одном дне", "steps_per_day", 8,    int,   lambda v: 1 <= v <= 100,       "нужно 1..100"),
    ("Seed (пусто = случайный)", "seed",          42,   None,  lambda v: True,                ""),
    ("Скорость анимации, дн/с",  "dps",           2.5,  float, lambda v: 0.01 <= v <= 500,    "нужно 0.01..500"),
    ("Дней за ручной шаг",       "step_days",     1.0,  float, lambda v: 0.01 <= v <= 100,    "нужно 0.01..100"),
    ("Кадр GIF каждые N кадров", "gif_every",     8,    int,   lambda v: v >= 1,              "нужно >= 1"),
    ("Масштаб GIF-кадра",        "gif_scale",     0.5,  float, lambda v: 0.1 <= v <= 1,       "нужно 0.1..1"),
]

def _tk_start_dialog(show=True):
    """Родное Tk-окно настроек (лёгкое: Entry/чекбоксы без matplotlib-виджетов).

    Возвращает cfg или None (окно закрыт / «Выход»).
    show=False — построить, но не показывать (для headless-тестов).
    """
    import tkinter as tk
    from tkinter import ttk

    out = {"cfg": None}
    try:
        root = tk.Tk()
    except tk.TclError:   # нет X/Tk окружения (совсем headless)
        return None
    root.title("Симуляция эпидемии — настройки")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=10)
    frm.grid(sticky="nsew")

    ttk.Label(frm, text="Настройки симуляции эпидемии (SEIRD)",
              font=("", 12, "bold")).grid(column=0, row=0, columnspan=4,
                                          sticky="w", pady=(0, 8))

    entries = {}
    half = (len(FIELDS) + 1) // 2
    for i, (lab, key, default, _cast, _check, _why) in enumerate(FIELDS):
        col_base = 0 if i < half else 2
        row = 1 + (i % half)
        ttk.Label(frm, text=lab).grid(column=col_base, row=row, sticky="e",
                                      padx=(0, 4), pady=2)
        ent = ttk.Entry(frm, width=12, justify="left")
        ent.insert(0, str(default))   # seed: пусто = случайный
        ent.grid(column=col_base + 1, row=row, sticky="w", pady=2)
        entries[key] = ent

    gif_var = tk.BooleanVar(value=DEFAULTS["gif"])
    dead_var = tk.BooleanVar(value=DEFAULTS["dead_infect"])
    gif_cb = ttk.Checkbutton(frm, text="Записывать GIF по шагам", variable=gif_var)
    dead_cb = ttk.Checkbutton(frm, text="Заражение от мёртвых", variable=dead_var)
    if not PIL_OK:
        gif_cb.state(["disabled"])   # без pillow GIF невозможен
        gif_var.set(False)
    gif_cb.grid(column=0, row=len(FIELDS) // 2 + 2, columnspan=2, sticky="w", pady=(10, 0))
    dead_cb.grid(column=2, row=len(FIELDS) // 2 + 2, columnspan=2, sticky="w", pady=(10, 0))

    mode_var = tk.IntVar(value=1)
    ttk.Radiobutton(frm, text="Авто-анимация", variable=mode_var, value=1).grid(
        column=0, row=len(FIELDS) // 2 + 3, columnspan=2, sticky="w")
    ttk.Radiobutton(frm, text="Ручные шаги", variable=mode_var, value=2).grid(
        column=2, row=len(FIELDS) // 2 + 3, columnspan=2, sticky="w")

    err_lbl = ttk.Label(frm, text="", foreground="red", wraplength=520)
    err_lbl.grid(column=0, row=len(FIELDS) // 2 + 4, columnspan=4, sticky="w", pady=(6, 0))

    def apply_values():
        """Собрать cfg из полей; ValueError с понятным текстом при ошибке."""
        cfg = {}
        for lab, key, _def, cast, check, why in FIELDS:
            raw = entries[key].get().strip().replace(",", ".")
            if key == "seed":
                cfg["seed"] = int(raw) if raw else None
                continue
            if not raw:
                raise ValueError(f"«{lab}»: пусто — введите число")
            try:
                val = cast(raw)
            except ValueError:
                raise ValueError(f"«{lab}»: не число ({raw})")
            if not check(val):
                raise ValueError(f"«{lab}»: {why}")
            cfg[key] = val
        if cfg["incub_lo"] > cfg["incub_hi"]:
            raise ValueError("Инкубация: мин больше макс")
        if cfg["ill_lo"] > cfg["ill_hi"]:
            raise ValueError("Болезнь: мин больше макс")
        if cfg["n_initial"] > cfg["n"]:
            raise ValueError("Нулевых пациентов больше численности населения")
        cfg["gif"] = bool(gif_var.get()) and PIL_OK
        cfg["dead_infect"] = bool(dead_var.get())
        cfg["mode"] = int(mode_var.get())
        return cfg

    def on_start(event=None):
        try:
            out["cfg"] = apply_values()
        except ValueError as e:
            err_lbl.config(text=f"ОШИБКА: {e}")
            return
        root.destroy()

    def on_quit(event=None):
        out["cfg"] = None
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_quit)
    btns = ttk.Frame(frm)
    btns.grid(column=0, row=len(FIELDS) // 2 + 5, columnspan=4, pady=(10, 0))
    ttk.Button(btns, text="Старт", command=on_start).grid(column=0, row=0, padx=6)
    ttk.Button(btns, text="Выход", command=on_quit).grid(column=1, row=0, padx=6)

    if show:
        print("Задайте параметры в окне и нажмите «Старт».")
        root.mainloop()
        return out["cfg"]
    # для headless-тестов
    out.update(root=root, entries=entries, on_start=on_start, on_quit=on_quit,
               err_lbl=err_lbl, dead_var=dead_var, gif_var=gif_var, mode_var=mode_var)
    return out


# ------------------------- модель -------------------------
class Epidemic:
    """Основная модель: позиции, состояния, шаг симуляции."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        n, w = cfg["n"], cfg["world"]

        self.pos = self.rng.random((n, 2)) * w
        self.state = np.zeros(n, dtype=np.int8)              # 0=S,1=E,2=I,3=R,4=D
        self.timer = np.zeros(n, dtype=np.int32)             # дней до смены состояния
        self.velocity = self.rng.normal(0.0, cfg["speed"] / 2.0, (n, 2))

        first = self.rng.choice(n, cfg["n_initial"], replace=False)
        self.state[first] = 2
        self.timer[first] = self.draw_duration(cfg["ill_lo"], cfg["ill_hi"], first.size)

        self.history = [self.bincount()]

    def bincount(self):
        return np.bincount(self.state, minlength=5)

    def _contact_matrix(self, sick, healthy):
        """Оригинальный путь: broadcasting-матрица расстояний (все H x I пар)."""
        cfg = self.cfg
        diff = self.pos[healthy][:, None, :] - self.pos[sick][None, :, :]
        return ((diff * diff).sum(-1) < cfg["radius"] ** 2).any(axis=1)

    def _contact_tree(self, sick, healthy):
        """Быстрый путь: ближайший больной в пределах радиуса (scipy cKDTree).

        Семантика та же: 'в радиусе есть хоть один больной'.
        """
        tree = cKDTree(self.pos[sick])
        dist, _ = tree.query(self.pos[healthy], k=1,
                             distance_upper_bound=self.cfg["radius"])
        return np.isfinite(dist)

    def draw_duration(self, lo, hi, size):
        if size == 0:
            return np.empty(0, dtype=np.int32)
        lo_i, hi_i = int(round(lo)), int(round(hi))
        if hi_i <= lo_i:
            return np.full(size, lo_i, dtype=np.int32)
        return self.rng.integers(lo_i, hi_i + 1, size=size)

    # --- внутренний шаг (1/steps_per_day дня) ---
    def step(self):
        cfg = self.cfg
        dt = 1.0 / cfg["steps_per_day"]

        # движение + отражение от границ
        self.pos = self.pos + self.velocity * dt
        mask = (self.pos < 0) | (self.pos > cfg["world"])
        self.velocity[mask] *= -1
        self.pos = np.clip(self.pos, 0.0, cfg["world"])

        # заражение здоровых темпами, приведёнными к длительности шага
        # (по флагу dead_infect источниками инфекции становятся и мёртвые)
        source_states = (2, 4) if cfg.get("dead_infect") else (2,)
        sick = np.where(np.isin(self.state, source_states))[0]
        healthy = np.where(self.state == 0)[0]
        if sick.size and healthy.size:
            # KD-дерево на больших матрицах контактов, иначе чистый broadcasting
            if SCIPY_OK and sick.size * healthy.size > KD_PAIRS_THRESHOLD:
                contact = self._contact_tree(sick, healthy)
            else:
                contact = self._contact_matrix(sick, healthy)
            p_step = 1.0 - (1.0 - cfg["infect_p"]) ** dt
            caught = healthy[contact & (self.rng.random(healthy.size) < p_step)]
            if caught.size:
                self.state[caught] = 1
                self.timer[caught] = self.draw_duration(
                    cfg["incub_lo"], cfg["incub_hi"], caught.size)

        # перевод таймеров: инкубация -> болезнь, болезнь -> выбытие/смерть
        to_ill = (self.state == 1) & (self.timer <= 0)
        self.state[to_ill] = 2
        self.timer[to_ill] = self.draw_duration(cfg["ill_lo"], cfg["ill_hi"], to_ill.sum())

        to_rem = np.where((self.state == 2) & (self.timer <= 0))[0]
        if to_rem.size:
            roll = self.rng.random(to_rem.size)
            dies = roll < cfg["lethality"]                    # погибшие
            self.state[to_rem[dies]] = 4
            self.velocity[to_rem[dies]] = 0.0                 # погибшие не движутся
            left = to_rem[~dies]
            survived_immune = left[self.rng.random(left.size) < cfg["immunity"]]
            self.state[survived_immune] = 3                   # иммунитет -> R
            # остальные потеряли иммунитет -> снова восприимчивы (S)
            rest = np.setdiff1d(left, survived_immune, assume_unique=False)
            self.state[rest] = 0
        self.timer[to_rem] = 0
        aging = np.isin(self.state, [1, 2])
        self.timer[aging] -= 1

    def advance(self, days):
        """Продвинуть симуляцию на `days` дней (float), вернуть int реально пройденных дней."""
        total = round(days * self.cfg["steps_per_day"])
        for k in range(total):
            self.step()
            if (k + 1) % self.cfg["steps_per_day"] == 0:
                self.history.append(self.bincount())
        return total // self.cfg["steps_per_day"]

    @property
    def day(self):
        return len(self.history) - 1

    @property
    def done(self):
        return self.day >= self.cfg["days"]


# ------------------------- визуализация -------------------------
class Viz:
    def __init__(self, sim):
        self.sim = sim
        self.fig, (self.ax_map, self.ax_inf) = plt.subplots(1, 2, figsize=(12, 5.5))
        self.fig.canvas.manager.set_window_title("Симуляция эпидемии")

        w = sim.cfg["world"]
        self.scat = self.ax_map.scatter(sim.pos[:, 0], sim.pos[:, 1], s=10,
                                        c=[COLORS[s] for s in sim.state])
        self.ax_map.set_xlim(0, w)
        self.ax_map.set_ylim(0, w)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_xticks([]), self.ax_map.set_yticks([])

        self.lines = {key: self.ax_inf.plot([], [], lw=2, color=c, label=lbl)[0]
                      for key, lbl, c in zip("SEIRD", LABELS, COLORS)}
        self.ax_inf.set_ylim(0, sim.cfg["n"])
        self.ax_inf.set_xlabel("Дни")
        self.ax_inf.set_ylabel("Число агентов")
        self.ax_inf.set_title("Динамика эпидемии")
        self.ax_inf.legend(loc="upper right")
        self.ax_inf.grid(alpha=0.3)
        self.refresh()

    def refresh(self):
        sim = self.sim
        self.scat.set_offsets(sim.pos)
        self.scat.set_color([COLORS[s] for s in sim.state])
        self.ax_map.set_title(f"Карта мира — день {sim.day}")

        hist = np.array(sim.history)
        t = np.arange(hist.shape[0])
        for k, key in enumerate("SEIRD"):
            self.lines[key].set_data(t, hist[:, k])
        self.ax_inf.set_xlim(0, max(20, sim.day + 5))
        self.fig.canvas.draw_idle()

    def set_status(self, text):
        """Вспомогательная строка в заголовке окна (скорость, пауза)."""
        try:
            self.fig.canvas.manager.set_window_title(
                f"Симуляция эпидемии — день {self.sim.day} | {text}")
        except Exception:
            pass  # заголовок не критичен при некоторых бекендах

    def capture_frame(self, scale=0.5):
        """Снимок окна в виде PIL Image (для GIF)."""
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        img = Image.fromarray(buf)
        if scale != 1.0:
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))))
        return img


# ------------------------- запись GIF и CSV -------------------------
class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = []
        self.counter = 0
        self.finalized = False
        self.enabled = cfg["gif"] and PIL_OK
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Каждому запуску — своя папка: sims\sim-<дата-время>\
        self.sim_dir = HERE / "sims" / f"sim-{stamp}"
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        self.gif_path = self.sim_dir / "epidemic.gif"
        self.csv_path = self.sim_dir / "epidemic.csv"
        self.plot_path = self.sim_dir / "epidemic_plot.png"

    def on_frame(self, viz):
        """Вызывать после каждого отображаемого кадра; GIF кадр — каждый N-й."""
        if not self.enabled:
            return
        self.counter += 1
        if self.counter % self.cfg["gif_every"] == 0:
            self.capture_now(viz)

    def capture_now(self, viz):
        """Принудительный кадр (клавиша s)."""
        if self.enabled:
            self.frames.append(viz.capture_frame(self.cfg["gif_scale"]))

    def save_gif(self):
        if not self.enabled or not self.frames:
            return None
        self.frames[0].save(
            self.gif_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=120,
            loop=0,
        )
        return self.gif_path

    def save_csv(self, history):
        hist = np.array(history)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("день,S,E,I,R,D\n")
            for day, row in enumerate(hist):
                f.write(f"{day},{int(row[0])},{int(row[1])},{int(row[2])},"
                        f"{int(row[3])},{int(row[4])}\n")
        return self.csv_path

    def save_plot(self, sim):
        """График динамики как отдельный файл PNG (фигура не та, что в окне)."""
        hist = np.array(sim.history)
        t = np.arange(hist.shape[0])
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (lbl, col) in enumerate(zip(LABELS, COLORS)):
            ax.plot(t, hist[:, i], lw=2, color=col, label=lbl)
        ax.set_xlim(0, max(20, sim.day + 1))
        ax.set_ylim(0, sim.cfg["n"])
        ax.set_xlabel("Дни")
        ax.set_ylabel("Число агентов")
        ax.set_title("Динамика эпидемии")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)
        return self.plot_path

    def finalize(self, sim):
        if self.finalized:
            return
        self.finalized = True
        gif = self.save_gif()
        csv = self.save_csv(sim.history)
        plot = self.save_plot(sim)
        c = sim.bincount()
        print("\n===== ИТОГИ СИМУЛЯЦИИ =====")
        print(f"  дней пройдено: {sim.day}")
        for lbl, cnt in zip(LABELS, c):
            print(f"  {lbl}: {int(cnt)}")
        print(f"  не заразилось: {int(c[0])} ({c[0] / sim.cfg['n']:.1%})")
        print(f"  папка прогона: {self.sim_dir}")
        if gif:
            print(f"  GIF:  {gif}  ({len(self.frames)} кадров)")
        print(f"  CSV:  {csv}")
        print(f"  график: {plot}")


# ------------------------- панель управления -------------------------
LOG_MIN_MANUAL, LOG_MAX_MANUAL = np.log10(0.01), np.log10(100.0)  # интервал: 0.01..100 дн

def _buttons_row(fig, labels, y=0.02, h=0.07):
    """Строка из n кнопок внизу фигуры. Возвращает список Button."""
    n = len(labels)
    left, right, gap = 0.02, 0.98, 0.012
    w = (right - left - (n - 1) * gap) / n
    btns = []
    for i, lbl in enumerate(labels):
        ax_b = fig.add_axes([left + i * (w + gap), y, w, h])
        btn = Button(ax_b, lbl)
        btn.label.set_fontsize(9)
        btns.append(btn)
    return btns

def _log_slider(fig, log_min, log_max, log_init):
    """Логарифмический слайдер во всю ширину над кнопками."""
    ax_s = fig.add_axes([0.02, 0.135, 0.96, 0.045])
    sl = Slider(ax_s, "", log_min, log_max, valinit=log_init)
    sl.label.set_fontsize(9)
    sl.valtext.set_visible(False)  # значение выводим в подписи сами
    return sl


# ------------------------- режимы запуска -------------------------
def _auto_controls(sim, viz, rec):
    """Создание панели авто-режима (кнопки; ход по реальному времени).

    Скорость задаётся параметром 'dps' (дней симуляции в секунду реального
    времени) — из окна настроек или клавишами +/-. Возвращает dict с колбэками
    и состоянием; run_auto довешивает анимацию. Тестируется headless.
    """
    spd = sim.cfg["steps_per_day"]
    fig = viz.fig
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.20)
    state = {"paused": False, "dps": sim.cfg["dps"], "steps": 0, "ani": None,
             "acc": 0.0, "t": None}   # t=None: часы ставятся на первом кадре

    def status_text():
        return f"{state['dps']:.3g} дней/с | {'ПАУЗА' if state['paused'] else 'идёт'}"

    btn_pause, btn_frame, btn_gif, btn_save, btn_quit = _buttons_row(
        fig, ["Пауза/Пуск", "Кадр вперёд", "GIF: вкл", "Сохранить GIF", "Выход"])

    def refresh_ui():
        btn_pause.label.set_text("Пуск" if state["paused"] else "Пауза")
        btn_gif.label.set_text("GIF: вкл" if rec.enabled else "GIF: выкл")
        viz.set_status(status_text())
        fig.canvas.draw_idle()

    def set_speed(dps_new):
        """Скорость хода: цель 'дней симуляции в секунду' (сам темпер не нужен)."""
        state["dps"] = min(500.0, max(0.01, dps_new))
        state["acc"] = 0.0
        refresh_ui()
        print(f"скорость: {state['dps']:.3g} дней/с"
              + (" | ПАУЗА" if state["paused"] else ""))

    def set_pause(paused):
        state["paused"] = paused
        state["t"] = time.perf_counter()   # не копить шаги за время паузы
        refresh_ui()
        print("ПАУЗА" if paused else "продолжили")

    def set_gif(enabled):
        rec.enabled = enabled
        refresh_ui()
        print("запись GIF:", "вкл" if enabled else "выкл")

    def _step_sub():
        """Один внутренний шаг + запись истории на границе дня."""
        sim.step()
        state["steps"] += 1
        if state["steps"] % spd == 0:
            sim.history.append(sim.bincount())
            return True
        return False

    def update(_frame=None):
        """Кадр анимации: продвинуть симуляцию на столько шагов, сколько
        прошло реального времени (дней/с * шагов/день * dt). Так скорость
        dps соблюдается, пока машина успевает, даже если рисование медленное.
        """
        if state["paused"] or sim.done:
            return
        now = time.perf_counter()
        if state["t"] is None:
            state["t"] = now   # первый кадр: только ставим часы, без долга
            return
        dt = max(0.0, min(now - state["t"], 0.5))
        state["t"] = now
        state["acc"] += state["dps"] * spd * dt
        todo = min(int(state["acc"]), 8192)
        state["acc"] -= todo
        day0 = sim.day
        for _ in range(todo):
            _step_sub()
            if sim.done:
                print(f"Симуляция завершена (день {sim.day}). "
                      "Окно можно закрыть («Выход»).")
                break
        if sim.day != day0:
            viz.set_status(status_text())
        viz.refresh()
        rec.on_frame(viz)

    def step_one_frame():
        if sim.done or state["steps"] >= sim.cfg["days"] * spd:
            print("Симуляция исчерпана — используйте «Выход».")
            return
        state["t"] = time.perf_counter()   # ручное действие перезапускает часы
        _step_sub()
        viz.refresh()
        rec.on_frame(viz)
        viz.set_status(status_text())

    def save_gif_now():
        rec.capture_now(viz)
        rec.save_gif()
        print(f"GIF сохранён: {rec.gif_path}")

    def on_key(event):
        key = event.key
        if key in ("space", "p", "з"):
            set_pause(not state["paused"])
        elif key == "right":
            step_one_frame()
        elif key in ("plus", "="):
            set_speed(state["dps"] * 2)
        elif key == "minus":
            set_speed(state["dps"] / 2)
        elif key in ("s", "ы"):
            save_gif_now()
        elif key in ("g", "п"):
            set_gif(not rec.enabled)
        elif key in ("q", "escape", "й"):
            plt.close(fig)

    btn_pause.on_clicked(lambda e: set_pause(not state["paused"]))
    btn_frame.on_clicked(lambda e: step_one_frame())
    btn_gif.on_clicked(lambda e: set_gif(not rec.enabled))
    btn_save.on_clicked(lambda e: save_gif_now())
    btn_quit.on_clicked(lambda e: plt.close(fig))

    return {"state": state, "refresh_ui": refresh_ui, "update": update,
            "on_key": on_key, "spd": spd,
            "buttons": {"pause": btn_pause, "frame": btn_frame, "gif": btn_gif,
                        "save": btn_save, "quit": btn_quit}}


def run_auto(sim, viz, rec):
    """Автоматическая анимация с кнопками: пауза, покадровый шаг, GIF.

    Кадры рисуются, как быстро успевает рисование; ход симуляции привязан к
    реальному времени (state['dps'] дней в секунду), поэтому перерисовка не
    влияет на скорость симуляции.
    """
    ctl = _auto_controls(sim, viz, rec)
    state, update = ctl["state"], ctl["update"]

    state["ani"] = animation.FuncAnimation(  # держим ссылку: не должна стать мусором
        viz.fig, update, interval=25, blit=False, cache_frame_data=False)
    viz.fig.canvas.mpl_connect("close_event", lambda e: rec.finalize(sim))
    viz.fig.canvas.mpl_connect("key_press_event", ctl["on_key"])
    ctl["refresh_ui"]()
    print(f"\nСкорость: {state['dps']:.3g} дней/с (клавиши +/- удваивают/делят; "
          "значение задаётся в окне настроек). Панель кнопок внизу окна; "
          "горячие клавиши тоже работают.")
    plt.show()
    # finalize подстрахован от двойного вызова
    rec.finalize(sim)
    return ctl


def _manual_controls(sim, viz, rec):
    """Создание панели ручного режима. Возвращает dict для run_manual/тестов."""
    fig = viz.fig
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.24)
    state = {"interval": float(sim.cfg["step_days"])}

    slider = _log_slider(fig, LOG_MIN_MANUAL, LOG_MAX_MANUAL, np.log10(state["interval"]))
    btn_step, btn_gif, btn_save, btn_quit = _buttons_row(
        fig, ["Шаг ->", "GIF: вкл", "Сохранить GIF", "Выход"])

    def refresh_ui():
        slider.label.set_text(f"Интервал шага: {state['interval']:.3g} дней")
        btn_gif.label.set_text("GIF: вкл" if rec.enabled else "GIF: выкл")
        fig.canvas.draw_idle()

    def set_interval(days_new, via_slider=False):
        state["interval"] = min(100.0, max(0.01, days_new))
        if not via_slider:
            slider.set_val(np.log10(state["interval"]))  # вызовет set_interval(..., True)
        refresh_ui()
        print(f"интервал шага = {state['interval']:.3g} дней")

    def on_step(event=None):
        if sim.done:
            print("Предел по дням достигнут — «Выход».")
            return
        sim.advance(state["interval"])
        viz.refresh()
        rec.on_frame(viz)

    def save_gif_now():
        rec.capture_now(viz)
        rec.save_gif()
        print(f"GIF сохранён: {rec.gif_path}")

    def set_gif(enabled):
        rec.enabled = enabled
        refresh_ui()
        print("запись GIF:", "вкл" if enabled else "выкл")

    def on_key(event):
        key = event.key
        if key in ("right", " ", "space"):
            if not sim.done:
                on_step()
        elif key in ("i", "ш"):
            print("Интервал шага настраивается слайдером на панели окна.")
        elif key in ("s", "ы"):
            save_gif_now()
        elif key in ("g", "п"):
            set_gif(not rec.enabled)
        elif key in ("q", "escape", "й"):
            plt.close(fig)

    slider.on_changed(lambda val: set_interval(10.0 ** val, via_slider=True))
    btn_step.on_clicked(on_step)
    btn_gif.on_clicked(lambda e: set_gif(not rec.enabled))
    btn_save.on_clicked(lambda e: save_gif_now())
    btn_quit.on_clicked(lambda e: plt.close(fig))

    return {"state": state, "slider": slider, "refresh_ui": refresh_ui,
            "on_key": on_key, "on_step": on_step,
            "buttons": {"step": btn_step, "gif": btn_gif,
                        "save": btn_save, "quit": btn_quit}}


def run_manual(sim, viz, rec):
    """Ручное управление: кнопки на панели + клавиши (Right/Space = шаг)."""
    ctl = _manual_controls(sim, viz, rec)
    viz.fig.canvas.mpl_connect("key_press_event", ctl["on_key"])
    viz.refresh()
    rec.on_frame(viz)  # нулевой кадр тоже попадает в GIF-хронологию
    ctl["refresh_ui"]()
    print("\nПанель кнопок внизу окна: Шаг, Интервал (слайдер), GIF, Выход. "
          "Клавиши Right/Space, i, s, g, q тоже работают.")
    plt.show()
    rec.finalize(sim)
    return ctl


# ------------------------- точка входа -------------------------
def main():
    cfg = _tk_start_dialog()
    if cfg is None:
        print("Настройки закрыты без запуска.")
        return
    sim = Epidemic(cfg)
    viz = Viz(sim)
    rec = Recorder(cfg)

    flags = []
    if cfg["lethality"]:
        flags.append(f"смертность {cfg['lethality']:.0%}")
    if cfg["dead_infect"]:
        flags.append("заражение от мёртвых ВКЛ")
    print(f"\nСтарт: {cfg['n']} агентов, seed={cfg['seed'] if cfg['seed'] is not None else 'случайный'},"
          + (", " + ", ".join(flags) if flags else ""))
    if cfg["mode"] == 1:
        run_auto(sim, viz, rec)
    else:
        run_manual(sim, viz, rec)


if __name__ == "__main__":
    main()
