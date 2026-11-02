"""Playable window for the JAX tile MMO. CPU only.

You control player 0 (the human). Other players are bots. Monsters chase and fight.

Opens fullscreen. Controls: WASD or arrow keys to move, Space to attack, R for a new world, Esc to quit.
Run from C:\\Users\\crp27\\jaxspike:  venv\\Scripts\\python.exe play.py
"""
import os
import sys
import tkinter as tk
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bots.policies import bot_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game.config import ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD, MAP, MONSTER_HP, PLAYER_MAX_HP  # noqa: E402
from game.step import step  # noqa: E402

TILE = 32            # pixels per tile (Tibia uses 32)
HUD_H = 40           # pixel strip under the map for the status lines
TICK_MS = 120        # game ticks per second is about 8
HUMAN = 0            # player slot the keyboard controls

# Move codes from config: 0 stay, 1 north (y-1), 2 south (y+1), 3 west (x-1), 4 east (x+1).
KEY_TO_MOVE = {
    "w": 1, "Up": 1,
    "s": 2, "Down": 2,
    "a": 3, "Left": 3,
    "d": 4, "Right": 4,
}

bot_inputs_jit = jax.jit(bot_inputs)
step_jit = jax.jit(step)


def unbatch(worlds):
    return jax.tree_util.tree_map(lambda x: x[0], worlds)


def draw_grass(c, sx, sy, shade):
    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill=("#2d4a24", "#31502a", "#294520")[shade], outline="")
    c.create_line(sx + 4, sy + TILE - 3, sx + 5, sy + TILE - 7, fill="#3f6b33")
    c.create_line(sx + 10, sy + TILE - 4, sx + 11, sy + TILE - 8, fill="#3f6b33")


def draw_wall(c, sx, sy):
    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill="#6b6b6b", outline="#444444")
    c.create_line(sx, sy + TILE // 2, sx + TILE, sy + TILE // 2, fill="#555555")
    c.create_line(sx + TILE // 2, sy, sx + TILE // 2, sy + TILE // 2, fill="#555555")
    c.create_line(sx + 2, sy + 2, sx + TILE - 3, sy + 2, fill="#8a8a8a")


def draw_rat(c, sx, sy):
    c.create_line(sx + TILE - 3, sy + TILE - 5, sx + TILE, sy + TILE - 10, fill="#c58b8b")
    c.create_oval(sx + 3, sy + 5, sx + TILE - 3, sy + TILE - 3, fill="#7a5b3a", outline="#3d2b1a")
    c.create_oval(sx + 1, sy + 5, sx + 7, sy + 11, fill="#7a5b3a", outline="#3d2b1a")
    c.create_oval(sx + 3, sy + 2, sx + 7, sy + 6, fill="#c58b8b", outline="")
    c.create_oval(sx + 2, sy + 7, sx + 4, sy + 9, fill="#ff3333", outline="")


def draw_orc(c, sx, sy):
    c.create_oval(sx + 2, sy + 3, sx + TILE - 2, sy + TILE - 1, fill="#4f8f3a", outline="#24431a")
    c.create_oval(sx + 4, sy + 6, sx + 7, sy + 8, fill="#ff3333", outline="")
    c.create_oval(sx + 9, sy + 6, sx + 12, sy + 8, fill="#ff3333", outline="")
    c.create_line(sx + 5, sy + 11, sx + 6, sy + 14, fill="#fff5d6", width=2)
    c.create_line(sx + 11, sy + 11, sx + 10, sy + 14, fill="#fff5d6", width=2)


def draw_player(c, sx, sy, body, outline, dead):
    if dead:
        body = "#555555"
    c.create_oval(sx + 4, sy + 1, sx + TILE - 4, sy + TILE // 2 + 2, fill="#f1c27d", outline="#7a5530")
    c.create_rectangle(sx + 3, sy + TILE // 2 + 2, sx + TILE - 3, sy + TILE - 2, fill=body, outline=outline)
    if dead:
        c.create_line(sx + 3, sy + 2, sx + TILE - 3, sy + TILE - 2, fill="#ff4444", width=2)
        c.create_line(sx + TILE - 3, sy + 2, sx + 3, sy + TILE - 2, fill="#ff4444", width=2)


def draw_drop(c, sx, sy, item):
    if item == ITEM_GOLD_COIN:
        c.create_oval(sx + 4, sy + 4, sx + TILE - 4, sy + TILE - 4, fill="#f5c542", outline="#a07d12", width=2)
        c.create_oval(sx + 7, sy + 7, sx + TILE - 7, sy + TILE - 7, outline="#a07d12")
    elif item == ITEM_MEAT:
        c.create_oval(sx + 3, sy + 5, sx + TILE - 3, sy + TILE - 4, fill="#c0392b", outline="#6e1d15")
        c.create_oval(sx + 11, sy + 3, sx + 14, sy + 6, fill="#f5e6c8", outline="")
    elif item == ITEM_SWORD:
        c.create_line(sx + 3, sy + TILE - 3, sx + TILE - 3, sy + 3, fill="#d7dde3", width=3)
        c.create_line(sx + 5, sy + TILE - 6, sx + 9, sy + TILE - 6, fill="#8b5a2b", width=3)
    else:
        c.create_oval(sx + 6, sy + 6, sx + TILE - 6, sy + TILE - 6, fill="#e8c547", outline="")


class Game:
    def __init__(self, root):
        self.root = root
        self.seed = 0
        self.held = []           # held movement keys, most recent last
        self.attacking = False
        self.paused = False
        # Fullscreen: as many whole tiles as fit, odd counts so the human sits on the centre tile.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.view_w = (sw // TILE) | 1
        self.view_h = ((sh - HUD_H) // TILE) | 1
        self.off_x = (sw - self.view_w * TILE) // 2  # centre the map strip on the screen
        self.canvas = tk.Canvas(root, width=sw, height=sh, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        root.bind("<KeyPress>", self.on_press)
        root.bind("<KeyRelease>", self.on_release)
        root.bind("<space>", lambda e: self.set_attack(True))
        root.bind("<KeyRelease-space>", lambda e: self.set_attack(False))
        root.bind("<r>", lambda e: self.reset())
        root.bind("<p>", lambda e: self.toggle_pause())
        root.bind("<Escape>", lambda e: root.destroy())
        self.reset()

    def reset(self):
        self.world = unbatch(make_world_batch(self.seed, 1))
        self.seed += 1
        self.tick_no = 0
        self.last_stats = (0, 0)
        self.draw()

    def toggle_pause(self):
        self.paused = not self.paused

    def set_attack(self, on):
        self.attacking = on

    def on_press(self, event):
        k = event.keysym
        if k in KEY_TO_MOVE and k not in self.held:
            self.held.append(k)

    def on_release(self, event):
        if event.keysym in self.held:
            self.held.remove(event.keysym)

    def human_move(self):
        if not self.held:
            return 0
        return KEY_TO_MOVE[self.held[-1]]

    def run_tick(self):
        if not self.paused:
            key = jax.random.PRNGKey(self.tick_no)
            inputs = bot_inputs_jit(self.world, key)
            move = np.array(inputs["move"])
            attack = np.array(inputs["attack"])
            move[HUMAN] = self.human_move()
            attack[HUMAN] = self.attacking
            inputs = {"move": jnp.asarray(move, jnp.int32), "attack": jnp.asarray(attack)}
            self.world = step_jit(self.world, inputs)
            self.tick_no += 1
            self.draw()
        self.root.after(TICK_MS, self.run_tick)

    def draw(self):
        w = jax.device_get(self.world)
        c = self.canvas
        c.delete("all")
        hx, hy = w["players"]["pos"][HUMAN]
        x0 = int(hx) - self.view_w // 2
        y0 = int(hy) - self.view_h // 2

        def to_screen(x, y):
            return self.off_x + (x - x0) * TILE, (y - y0) * TILE

        walls = w["walls"]
        for ty in range(self.view_h):
            for tx in range(self.view_w):
                x, y = x0 + tx, y0 + ty
                sx, sy = to_screen(x, y)
                if not (0 <= x < MAP and 0 <= y < MAP):
                    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill="#000000", outline="")
                elif walls[x, y]:
                    draw_wall(c, sx, sy)
                else:
                    draw_grass(c, sx, sy, (x * 7 + y * 13) % 3)

        d = w["drops"]
        def on_screen(x, y):
            return x0 <= x < x0 + self.view_w and y0 <= y < y0 + self.view_h

        for i in np.flatnonzero(d["active"]):
            x, y = d["pos"][i]
            if on_screen(x, y):
                draw_drop(c, *to_screen(x, y), int(d["item"][i]))

        m = w["monsters"]
        for i in np.flatnonzero(m["active"]):
            x, y = m["pos"][i]
            if not on_screen(x, y):
                continue
            sx, sy = to_screen(x, y)
            if m["kind"][i] == 0:
                draw_rat(c, sx, sy)
            else:
                draw_orc(c, sx, sy)
            hp_frac = max(m["hp"][i], 0) / MONSTER_HP[int(m["kind"][i])]
            c.create_rectangle(sx + 2, sy, sx + 2 + (TILE - 4) * hp_frac, sy + 2, fill="#ff4444", outline="")

        p = w["players"]
        for i in np.flatnonzero(p["active"]):
            x, y = p["pos"][i]
            if not on_screen(x, y):
                continue
            sx, sy = to_screen(x, y)
            if i == HUMAN:
                draw_player(c, sx, sy, "#ffd166", "#000000", bool(p["dead"][i]))
            else:
                draw_player(c, sx, sy, "#5dade2", "#1b4f72", bool(p["dead"][i]))

        hp = int(p["hp"][HUMAN])
        kills = int(p["kills"][HUMAN])
        deaths = int(p["deaths"][HUMAN])
        items = int(np.count_nonzero(p["inv"][HUMAN]))
        status = "PAUSED  " if self.paused else ""
        if p["dead"][HUMAN]:
            status += "YOU ARE DEAD (respawning) "
        hud_y = self.view_h * TILE
        c.create_text(
            8, hud_y + 12, anchor="w", fill="#dddddd", font=("Consolas", 11),
            text=f"HP {hp}/{PLAYER_MAX_HP}   kills {kills}   deaths {deaths}   items {items}   tick {self.tick_no}",
        )
        c.create_text(
            8, hud_y + 30, anchor="w", fill="#aaaaaa", font=("Consolas", 10),
            text=status + "WASD/arrows move  Space attack  P pause  R new world  Esc quit",
        )


def main():
    root = tk.Tk()
    root.title("JAX tile MMO demo")
    root.attributes("-fullscreen", True)
    game = Game(root)
    root.focus_force()
    root.after(TICK_MS, game.run_tick)
    root.mainloop()


if __name__ == "__main__":
    main()
