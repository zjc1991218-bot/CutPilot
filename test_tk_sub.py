import subprocess
def ask_save_path_via_subprocess(default_filename):
    script = f"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
file_path = filedialog.asksaveasfilename(initialfile="{default_filename}", title="选择保存位置")
print(file_path)
"""
    result = subprocess.run(["python3", "-c", script], capture_output=True, text=True)
    return result.stdout.strip()

# print(ask_save_path_via_subprocess("test.txt"))
