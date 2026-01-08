import sys
import os
import tkinter as tk
from tkinter import ttk

import scenario_gauntlet

BASE_DIR = os.path.dirname(__file__)

LOGO_PATH = os.path.join(BASE_DIR, "Kessler_Game600_x_600_px.png")


def start_game(root):
    root.destroy()
    scenario_gauntlet.main()


def quit_game(root):
    root.destroy()
    sys.exit()


def main():
    root = tk.Tk()
    root.title("Kessler Game")

    root.geometry("1000x1000")
    root.configure(bg="black")
    root.resizable(False, False)

    main_frame = tk.Frame(root, bg="black")
    main_frame.pack(expand=True)

    print("Loading logo from:", LOGO_PATH)

    # load logo
    try:
        logo_img = tk.PhotoImage(file=LOGO_PATH)
    except Exception as e:
        print("ERROR loading image:", e)
        logo_img = None

    if logo_img:
        logo_label = tk.Label(main_frame, image=logo_img, bg="black")
        logo_label.image = logo_img
        logo_label.pack(pady=(40, 30))
    else:
        fallback = tk.Label(
            main_frame,
            text="KESSLER GAME",
            font=("Arial", 44, "bold"),
            fg="white",
            bg="black",
        )
        fallback.pack(pady=(60, 40))

    style = ttk.Style()
    style.configure("Menu.TButton", font=("Arial", 22), padding=12)

    play_button = ttk.Button(
        main_frame,
        text="Play",
        style="Menu.TButton",
        command=lambda: start_game(root),
    )
    play_button.pack(pady=20, ipadx=25, ipady=10)

    quit_button = ttk.Button(
        main_frame,
        text="Quit",
        style="Menu.TButton",
        command=lambda: quit_game(root),
    )
    quit_button.pack(pady=10, ipadx=25, ipady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
