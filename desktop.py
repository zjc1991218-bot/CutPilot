import os
import sys
import time
import socket
import subprocess
import threading
try:
    import webview
except ImportError:
    print("正在安装依赖 pywebview...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pywebview"])
    import webview

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_streamlit(port):
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    # Hide the "Made with Streamlit" menu if possible
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], env=env)

def main():
    port = get_free_port()
    print(f"正在后台启动 Streamlit 引擎 (端口: {port})...")
    
    t = threading.Thread(target=run_streamlit, args=(port,), daemon=True)
    t.start()
    
    # 给 streamlit 几秒钟的启动时间
    time.sleep(3) 
    
    # 创建桌面窗口
    window = webview.create_window(
        "CutPilot Pro v1.3 - AI 数字化场记与粗剪工作台", 
        f"http://localhost:{port}",
        width=1280,
        height=800,
        min_size=(1024, 768)
    )
    
    print("启动桌面窗口...")
    webview.start()

if __name__ == '__main__':
    main()
