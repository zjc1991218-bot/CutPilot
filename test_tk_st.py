import streamlit as st
import tkinter as tk
from tkinter import filedialog

if st.button("Ask"):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.asksaveasfilename(
        initialfile="test.txt",
        title="选择保存位置",
    )
    root.destroy()
    st.write("Path:", file_path)
