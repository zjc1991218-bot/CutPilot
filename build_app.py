import PyInstaller.__main__
import os
import streamlit

streamlit_path = os.path.dirname(streamlit.__file__)

PyInstaller.__main__.run([
    'desktop.py',
    '--name=CutPilotPro',
    '--windowed',
    '--clean',
    '--noconfirm',
    f'--add-data={streamlit_path}:streamlit',
    '--add-data=app.py:.',
    '--hidden-import=streamlit',
    '--hidden-import=pywebview',
    '--hidden-import=pandas',
    '--hidden-import=docx',
    '--hidden-import=openai',
    '--hidden-import=streamlit.runtime.scriptrunner.magic_funcs'
])
