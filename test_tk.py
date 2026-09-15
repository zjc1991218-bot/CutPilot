import tkinter as tk
from tkinter import filedialog

def ask():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.asksaveasfilename(
        initialfile="test.txt",
        title="选择保存位置",
    )
    root.destroy()
    print("Path:", file_path)

ask()
