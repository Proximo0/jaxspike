"""Playable window for the JAX tile MMO: a network client for net/server.py. CPU only.

Opens fullscreen. Controls: WASD or arrow keys to move, Space to attack, E to eat, left click to walk to a
tile, Esc to quit.
Run from C:\\Users\\crp27\\jaxspike:  venv\\Scripts\\python.exe play.py --host 127.0.0.1 --port 8765 --name you
"""
import argparse
import getpass
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game.config import (  # noqa: E402
    ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD, MONSTER_HP,
    T_FLOOR, T_GRASS, T_ROAD, T_ROCK, T_TEMPLE, T_TREE, T_WALL, T_WATER,
)
from net.client import NetClient  # noqa: E402

TILE = 32            # pixels per tile (Tibia uses 32)
HUD_H = 40           # pixel strip under the map for the status lines

# Move codes from config: 0 stay, 1 north (y-1), 2 south (y+1), 3 west (x-1), 4 east (x+1).
KEY_TO_MOVE = {
    "w": 1, "Up": 1,
    "s": 2, "Down": 2,
    "a": 3, "Left": 3,
    "d": 4, "Right": 4,
}

# Flat fill colour per terrain id. None means a drawn tile (grass, tree, wall) handled in draw().
TERRAIN_FILL = {
    T_GRASS: None,
    T_ROAD: "#b89b6a",
    T_FLOOR: "#8a7a66",
    T_TEMPLE: "#e6d28a",
    T_WATER: "#2f6fb3",
    T_TREE: None,
    T_WALL: None,
    T_ROCK: "#3a3a3a",
}


def draw_grass(c, sx, sy, shade):
    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill=("#2d4a24", "#31502a", "#294520")[shade], outline="")
    c.create_line(sx + 4, sy + TILE - 3, sx + 5, sy + TILE - 7, fill="#3f6b33")
    c.create_line(sx + 10, sy + TILE - 4, sx + 11, sy + TILE - 8, fill="#3f6b33")


def draw_tree(c, sx, sy, shade):
    draw_grass(c, sx, sy, shade)
    c.create_oval(sx + 3, sy + 3, sx + TILE - 3, sy + TILE - 3, fill="#1f4d1a", outline="")


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
    def __init__(self, root, client):
        self.root = root
        self.client = client
        self.welcome = client.welcome
        self.held = []           # held movement keys, most recent last
        self.attacking = False
        self.eating = False
        self.sent = (0, False, False)
        self.terrain = None      # last non-null snap["terrain"]
        self.snap = None
        self.x0 = self.y0 = 0    # world coordinates of the top-left drawn tile
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        # The server fixes the view size; draw only the tiles that fit in the window.
        self.view_w = min(self.welcome["view_w"], (sw // TILE) | 1)
        self.view_h = min(self.welcome["view_h"], ((sh - HUD_H) // TILE) | 1)
        self.off_x = (sw - self.view_w * TILE) // 2  # centre the map strip on the screen
        self.canvas = tk.Canvas(root, width=sw, height=sh, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        root.bind("<KeyPress>", self.on_press)
        root.bind("<KeyRelease>", self.on_release)
        root.bind("<space>", lambda e: self.set_attack(True))
        root.bind("<KeyRelease-space>", lambda e: self.set_attack(False))
        root.bind("<e>", lambda e: self.set_eat(True))
        root.bind("<KeyRelease-e>", lambda e: self.set_eat(False))
        root.bind("<Button-1>", self.on_click)
        root.bind("<Escape>", lambda e: self.quit())

    def quit(self):
        self.client.close()
        self.root.destroy()

    def set_attack(self, on):
        self.attacking = on
        self.push_input()

    def set_eat(self, on):
        self.eating = on
        self.push_input()

    def on_press(self, event):
        k = event.keysym
        if k in KEY_TO_MOVE and k not in self.held:
            self.held.append(k)
            self.push_input()

    def on_release(self, event):
        if event.keysym in self.held:
            self.held.remove(event.keysym)
            self.push_input()

    def human_move(self):
        if not self.held:
            return 0
        return KEY_TO_MOVE[self.held[-1]]

    def push_input(self):
        cur = (self.human_move(), self.attacking, self.eating)
        if cur != self.sent:
            self.sent = cur
            self.client.send_input(*cur)

    def on_click(self, event):
        if self.snap is None:
            return
        tx = (event.x - self.off_x) // TILE
        ty = event.y // TILE
        if 0 <= tx < self.view_w and 0 <= ty < self.view_h:
            self.client.walk_to(self.x0 + tx, self.y0 + ty)

    def frame(self):
        snap = self.client.latest()
        if snap is not None:
            self.snap = snap
            if snap.get("terrain") is not None:
                self.terrain = snap["terrain"]
        self.draw()
        self.root.after(self.welcome["tick_ms"], self.frame)

    def draw(self):
        c = self.canvas
        c.delete("all")
        hud_y = self.view_h * TILE
        snap = self.snap
        if snap is None or self.terrain is None:
            c.create_text(8, hud_y + 12, anchor="w", fill="#dddddd", font=("Consolas", 11), text="waiting for server...")
            return
        you = snap["you"]
        hx, hy = you["pos"]
        self.x0 = hx - self.view_w // 2
        self.y0 = hy - self.view_h // 2
        x0, y0 = self.x0, self.y0

        def to_screen(x, y):
            return self.off_x + (x - x0) * TILE, (y - y0) * TILE

        def on_screen(x, y):
            return x0 <= x < x0 + self.view_w and y0 <= y < y0 + self.view_h

        t = self.terrain
        tx0, ty0, rows = t["x0"], t["y0"], t["rows"]
        for ty in range(self.view_h):
            for tx in range(self.view_w):
                x, y = x0 + tx, y0 + ty
                sx, sy = to_screen(x, y)
                i, j = x - tx0, y - ty0
                if not (0 <= i < len(rows) and 0 <= j < len(rows[0])):
                    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill="#000000", outline="")
                    continue
                tid = rows[i][j]
                fill = TERRAIN_FILL.get(tid)
                if fill is not None:
                    c.create_rectangle(sx, sy, sx + TILE, sy + TILE, fill=fill, outline="")
                elif tid == T_WALL:
                    draw_wall(c, sx, sy)
                elif tid == T_TREE:
                    draw_tree(c, sx, sy, (x * 7 + y * 13) % 3)
                else:
                    draw_grass(c, sx, sy, (x * 7 + y * 13) % 3)

        ents = snap["ents"]
        for e in ents:
            if e["kind"] == "d" and on_screen(*e["pos"]):
                draw_drop(c, *to_screen(*e["pos"]), e["item"])
        for e in ents:
            if e["kind"] != "m" or not on_screen(*e["pos"]):
                continue
            sx, sy = to_screen(*e["pos"])
            if e["mkind"] == 0:
                draw_rat(c, sx, sy)
            else:
                draw_orc(c, sx, sy)
            hp_frac = max(e["hp"], 0) / MONSTER_HP[e["mkind"]]
            c.create_rectangle(sx + 2, sy, sx + 2 + (TILE - 4) * hp_frac, sy + 2, fill="#ff4444", outline="")
        for e in ents:
            if e["kind"] != "p" or not on_screen(*e["pos"]) or list(e["pos"]) == [hx, hy]:
                continue
            sx, sy = to_screen(*e["pos"])
            draw_player(c, sx, sy, "#5dade2", "#1b4f72", False)
            c.create_text(sx + TILE // 2, sy - 6, fill="#ffffff", font=("Consolas", 9), text=e["name"])
        sx, sy = to_screen(hx, hy)
        draw_player(c, sx, sy, "#ffd166", "#000000", you["hp"] <= 0)

        items = sum(1 for i in you["inv"] if i)
        notice = f"   [{snap['notice']}]" if snap.get("notice") else ""
        c.create_text(
            8, hud_y + 12, anchor="w", fill="#dddddd", font=("Consolas", 11),
            text=f"HP {you['hp']}/{you['max_hp']}   kills {you['kills']}   deaths {you['deaths']}   items {items}"
            f"   town {you['town']}" + ("   SAFE" if you["safe"] else "") + notice,
        )
        down = not self.client.connected
        c.create_text(
            8, hud_y + 30, anchor="w", fill="#ff6666" if down else "#aaaaaa", font=("Consolas", 10),
            text="DISCONNECTED" if down else "WASD/arrows move  Space attack  E eat  click to walk  Esc quit",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--name", default=getpass.getuser().lower())
    args = ap.parse_args()
    client = NetClient(args.host, args.port, args.name)
    client.start()
    t0 = time.time()
    while client.welcome is None and client.error is None and time.time() - t0 < 10:
        time.sleep(0.02)
    if client.welcome is None:
        sys.exit(f"could not join: {client.error or 'timed out'}")
    root = tk.Tk()
    root.title("JAX tile MMO - " + args.name)
    root.attributes("-fullscreen", True)
    game = Game(root, client)
    root.focus_force()
    root.after(client.welcome["tick_ms"], game.frame)
    root.mainloop()
    client.close()


if __name__ == "__main__":
    main()
