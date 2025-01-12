import tkinter as tk
from tkinter import ttk
import subprocess

class AppController:
    def __init__(self, root):
        self.root = root
        self.root.title("Security System Controller")
        self.root.geometry("300x200")
        self.process = None

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(expand=True)

        self.start_button = ttk.Button(main_frame, text="Start App", command=self.start_app, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = ttk.Button(main_frame, text="Stop App", command=self.stop_app, state="disabled")
        self.stop_button.grid(row=1, column=0, padx=10, pady=10)

        self.restart_button = ttk.Button(main_frame, text="Restart App", command=self.restart_app, state="disabled")
        self.restart_button.grid(row=2, column=0, padx=10, pady=10)

    def start_app(self):
        if self.process is None:
            self.process = subprocess.Popen([".env\Scripts\python.exe", "run.py"]) 
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.restart_button.config(state="normal")

    def stop_app(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.restart_button.config(state="disabled")

    def restart_app(self):
        self.stop_app()
        self.start_app()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppController(root)
    root.mainloop()