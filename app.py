import os
import sys
import re
import io
import json
import time
import base64
import tempfile
import subprocess
import shutil
import math
import difflib
import urllib.request
import concurrent.futures
from html import escape
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from docx import Document

class TqdmProgressCapture:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.buf = ""
        self.original_stderr = sys.stderr
        self.max_pct = 0
        # Matches patterns like "Downloading [model.safetensors]:  45%|"
        self.pct_re = re.compile(r'\[(.*?)\]:\s*(\d+)%\|')

    def write(self, x):
        self.original_stderr.write(x)
        self.buf += x
        if '\r' in self.buf or '\n' in self.buf:
            lines = self.buf.replace('\r', '\n').split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                match = self.pct_re.search(line)
                if match:
                    filename = match.group(1)
                    pct = int(match.group(2))
                    
                    # Usually the model weight file takes the longest. 
                    # We only track progress of large files like .safetensors, .bin, .pt
                    if any(ext in filename for ext in ['.safetensors', '.bin', '.pt', 'model']):
                        if pct > self.max_pct:
                            self.max_pct = pct
                            try:
                                self.progress_bar.progress(self.max_pct / 100.0, text=f"{self.max_pct}%")
                            except Exception:
                                pass
            self.buf = ""

    def flush(self):
        self.original_stderr.flush()


# ----------------- 专属 Logo 图标探测 -----------------
LOGO_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "assets", "logo.png"),
    os.path.join(os.path.dirname(__file__), "logo.png"),
    "/Users/zhangjiacheng/Desktop/projects/video_cutter/assets/logo.png",
    "/Users/zhangjiacheng/Desktop/CutPilot.app/Contents/Resources/logo.png",
    os.path.expanduser("~/Desktop/CutPilot_Logo.png"),
]
APP_LOGO_PATH = next((p for p in LOGO_CANDIDATES if os.path.exists(p)), None)
APP_LOGO_IMAGE = Image.open(APP_LOGO_PATH) if APP_LOGO_PATH else "✂️"

def get_logo_base64():
    if APP_LOGO_PATH and os.path.exists(APP_LOGO_PATH):
        try:
            with open(APP_LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""

LOGO_B64 = get_logo_base64()

# ----------------- 页面基础配置 -----------------
st.set_page_config(
    page_title="CutPilot AI v1.6 - 数字化场记与粗剪", 
    layout="wide", 
    page_icon=APP_LOGO_IMAGE,
    initial_sidebar_state="expanded"
)

# ----------------- 苹果官方标准应用支持与模型存储区 -----------------
APP_SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/CutPilot")
APP_MODELS_DIR = os.path.join(APP_SUPPORT_DIR, "models")
try:
    os.makedirs(APP_MODELS_DIR, exist_ok=True)
except Exception:
    pass

# 全量重定向第三方 AI 下载引擎的落盘路径至苹果官方标准缓存区
os.environ["MODELSCOPE_CACHE"] = os.path.join(APP_MODELS_DIR, "modelscope")
os.environ["HF_HOME"] = os.path.join(APP_MODELS_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(APP_MODELS_DIR, "torch")

# ----------------- 配置文件与 API Key 本地持久化 -----------------
CONFIG_DIR = os.path.expanduser("~/.cutpilot")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_app_config():
    default_cfg = {
        "engine_type": "local",
        "dashscope_api_key": "",
        "dashscope_model": "paraformer-v2",
        "volc_api_key": "",
        "volc_app_id": "",
        "volc_resource_id": "volc.bigasr.auc_turbo"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg

def save_app_config(cfg):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

if "app_config" not in st.session_state:
    st.session_state["app_config"] = load_app_config()
app_cfg = st.session_state["app_config"]

# ----------------- 本地模型与多源镜像管理中心 -----------------
DOWNLOAD_MIRRORS = [
    {
        "id": "modelscope",
        "name": "阿里魔搭 ModelScope (国内官方高速 CDN)",
        "ping_url": "https://www.modelscope.cn",
        "type": "modelscope"
    },
    {
        "id": "hf_mirror",
        "name": "HF-Mirror (国内顶级高速镜像站)",
        "ping_url": "https://hf-mirror.com",
        "type": "huggingface",
        "endpoint": "https://hf-mirror.com"
    },
    {
        "id": "huggingface",
        "name": "Hugging Face (官方海外源)",
        "ping_url": "https://huggingface.co",
        "type": "huggingface",
        "endpoint": "https://huggingface.co"
    }
]

DOWNLOADABLE_RESOURCES = {
    "qwen3_asr_17b": {
        "id": "qwen3_asr_17b",
        "title": "🎬 Qwen3-ASR-1.7B (高精度主力版)",
        "desc": "影视级高精度台词识别，专业术语、同音字纠偏极佳，显存占用约 3.8GB",
        "size_estimate": "约 3.6 GB",
        "ms_model_id": "Qwen/Qwen3-ASR-1.7B",
        "hf_model_id": "Qwen/Qwen3-ASR-1.7B",
        "category": "ASR 语音引擎",
        "min_expected_size_mb": 1500
    },
    "qwen3_asr_06b": {
        "id": "qwen3_asr_06b",
        "title": "⚡ Qwen3-ASR-0.6B (极速轻量版)",
        "desc": "轻量级语音大模型，适合日常采访与短视频草稿速出，显存占用约 1.5GB",
        "size_estimate": "约 1.3 GB",
        "ms_model_id": "Qwen/Qwen3-ASR-0.6B",
        "hf_model_id": "Qwen/Qwen3-ASR-0.6B",
        "category": "ASR 语音引擎",
        "min_expected_size_mb": 600
    },
    "qwen3_aligner_06b": {
        "id": "qwen3_aligner_06b",
        "title": "🎯 Qwen3-ForcedAligner-0.6B (字级时码对齐引擎)",
        "desc": "高精度音素级强制对齐模型，用于毫秒级时码咬合与逐字规整校准",
        "size_estimate": "约 1.2 GB",
        "ms_model_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "hf_model_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "category": "音素对齐引擎",
        "min_expected_size_mb": 500
    }
}

def _calc_dir_size_and_validity(path, min_mb=500):
    try:
        total = 0
        has_weights = False
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    sz = os.path.getsize(fp)
                    total += sz
                    if f.endswith((".safetensors", ".bin", ".pt")):
                        has_weights = True
        return total, (has_weights and total >= min_mb * 1024 * 1024)
    except Exception:
        return 0, False

def find_local_resource_dir(res_info):
    """检测本地是否已存在该资源（优先探测苹果官方标准支持目录，兼容历史探测）"""
    ms_id = res_info.get("ms_model_id", "")
    hf_id = res_info.get("hf_model_id", "")
    min_mb = res_info.get("min_expected_size_mb", 500)
    
    candidates = []
    # 1. 优先探测苹果官方标准存储区 ~/Library/Application Support/CutPilot/models
    if ms_id:
        sanitized_ms = ms_id.replace("/", "--")
        candidates.append(os.path.join(APP_MODELS_DIR, "modelscope", "models", sanitized_ms))
        candidates.append(os.path.join(APP_MODELS_DIR, "modelscope", "hub", ms_id))
        candidates.append(os.path.join(APP_MODELS_DIR, "modelscope", "models", ms_id))
    if hf_id:
        sanitized_hf = "models--" + hf_id.replace("/", "--")
        candidates.append(os.path.join(APP_MODELS_DIR, "huggingface", "hub", sanitized_hf))
    candidates.append(os.path.join(APP_MODELS_DIR, res_info.get("id", "")))

    # 2. 兼容历史缓存路径
    if ms_id:
        sanitized_ms = ms_id.replace("/", "--")
        candidates.append(os.path.expanduser(f"~/.cache/modelscope/models/{sanitized_ms}"))
        candidates.append(os.path.expanduser(f"~/.cache/modelscope/hub/{ms_id}"))
        candidates.append(os.path.expanduser(f"~/.cache/modelscope/models/{ms_id}"))
    if hf_id:
        sanitized_hf = "models--" + hf_id.replace("/", "--")
        candidates.append(os.path.expanduser(f"~/.cache/huggingface/hub/{sanitized_hf}"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", res_info.get("id", "")))

    for base_p in candidates:
        if not os.path.isdir(base_p):
            continue
        snap_dir = os.path.join(base_p, "snapshots")
        if os.path.isdir(snap_dir):
            for sub in os.listdir(snap_dir):
                sub_p = os.path.join(snap_dir, sub)
                if os.path.isdir(sub_p):
                    sz, is_valid = _calc_dir_size_and_validity(sub_p, min_mb)
                    if is_valid:
                        return sub_p, sz
        sz, is_valid = _calc_dir_size_and_validity(base_p, min_mb)
        if is_valid:
            return base_p, sz

    return None, 0

def format_file_size(size_bytes):
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"

def get_dir_total_size(path):
    """计算目录总大小"""
    if not os.path.exists(path):
        return 0
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total

def test_single_mirror_latency(mirror):
    url = mirror["ping_url"]
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CutPilot-SpeedTest/1.2"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            latency = int((time.perf_counter() - t0) * 1000)
            return {**mirror, "latency": latency, "status": "online"}
    except Exception as e:
        return {**mirror, "latency": 9999, "status": "offline", "error": str(e)}

def get_sorted_mirror_speed_results():
    """并发测试各大镜像源并按延迟升序返回"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(DOWNLOAD_MIRRORS)) as executor:
        futures = [executor.submit(test_single_mirror_latency, m) for m in DOWNLOAD_MIRRORS]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda x: x["latency"])
    return results

def execute_resource_download(res_key, chosen_mirror_id=None):
    """根据指定或最优镜像源执行模型下载至苹果官方标准存储目录"""
    res_info = DOWNLOADABLE_RESOURCES.get(res_key)
    if not res_info:
        raise ValueError(f"未知下载项目: {res_key}")
    
    if not chosen_mirror_id or chosen_mirror_id == "auto":
        speeds = get_sorted_mirror_speed_results()
        chosen_mirror_id = speeds[0]["id"] if speeds and speeds[0]["status"] == "online" else "modelscope"

    if chosen_mirror_id == "modelscope":
        from modelscope import snapshot_download
        ms_id = res_info["ms_model_id"]
        ms_cache = os.path.join(APP_MODELS_DIR, "modelscope")
        return snapshot_download(ms_id, cache_dir=ms_cache)
    else:
        endpoint = "https://hf-mirror.com" if chosen_mirror_id == "hf_mirror" else "https://huggingface.co"
        os.environ["HF_ENDPOINT"] = endpoint
        from huggingface_hub import snapshot_download
        hf_id = res_info["hf_model_id"]
        hf_cache = os.path.join(APP_MODELS_DIR, "huggingface")
        return snapshot_download(repo_id=hf_id, cache_dir=hf_cache)

# ----------------- 阿里百炼与火山引擎连通性检测 -----------------
def test_dashscope_key(api_key):
    import requests
    try:
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
        res = requests.get(url, headers={"Authorization": f"Bearer {api_key.strip()}"}, timeout=8)
        if res.status_code == 200:
            return True, "✅ 阿里百炼 API Key 验证通过！"
        else:
            return False, f"❌ 验证未通过 (HTTP {res.status_code})"
    except Exception as e:
        return False, f"❌ 连接超时或网络异常: {str(e)[:80]}"

def test_volcengine_key(api_key, app_id, resource_id):
    import requests
    import uuid
    try:
        url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
        headers = {
            "X-Api-Key": api_key.strip(),
            "X-Api-Resource-Id": (resource_id or "volc.bigasr.auc_turbo").strip(),
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json"
        }
        if app_id and app_id.strip():
            headers["X-Api-App-Key"] = app_id.strip()
        res = requests.post(url, headers=headers, json={"user": {"uid": "test"}, "audio": {"format": "wav", "data": ""}}, timeout=8)
        status = res.headers.get("X-Api-Status-Code", "")
        msg = res.headers.get("X-Api-Message", "")
        if status == "20000000" or "empty audio" in msg.lower() or "audio data is empty" in msg.lower():
            return True, "✅ 火山引擎 API Key 鉴权验证通过！"
        elif status:
            return False, f"❌ 火山引擎鉴权提示 ({status}): {msg}"
        else:
            return False, f"❌ 验证未通过 (HTTP {res.status_code})"
    except Exception as e:
        return False, f"❌ 连接超时或网络异常: {str(e)[:80]}"

# ----------------- 侧边栏设置与外观主题 -----------------
with st.sidebar:
    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:68px; height:68px; border-radius:15px; margin-bottom:8px; box-shadow: 0 4px 14px rgba(59,130,246,0.35);">' if LOGO_B64 else ''
    st.markdown(f"""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            {logo_html}
            <h2 style="margin:0; font-weight:800; font-size:1.4rem;">CutPilot <span style="color:#3b82f6;">Pro</span> <span style="font-size:0.85rem; background:rgba(59,130,246,0.2); color:#60a5fa; padding:2px 7px; border-radius:6px; font-weight:600; margin-left:4px;">v1.6</span></h2>
            <p style="font-size:0.82rem; margin-top:5px; opacity:0.8;">本地大模型 & 阿里百炼/火山引擎</p>
        </div>
        <hr style="opacity:0.15; margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;'>🎨 界面外观主题</p>", unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "界面外观",
        ["深色沉浸 (推荐后期剪辑)", "浅色明亮 (清新护眼)", "跟随系统外观"],
        index=0,
        key="app_theme_mode",
        label_visibility="collapsed",
        help="剪辑工程默认推荐深色沉浸。若系统切换为白天浅色，可切换浅色或跟随系统。"
    )

    st.markdown("<br><p style='font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;'>🤖 语音引擎配置</p>", unsafe_allow_html=True)
    
    engine_type_names = {
        "local": "💻 本地离线引擎 (Qwen3-ASR)",
        "dashscope": "☁️ 阿里百炼 (DashScope 语音识别)",
        "volcengine": "🌋 火山引擎 (豆包语音大模型)"
    }
    engine_keys = list(engine_type_names.keys())
    saved_engine = app_cfg.get("engine_type", "local")
    if saved_engine not in engine_keys:
        saved_engine = "local"
    current_engine_idx = engine_keys.index(saved_engine)
    
    selected_engine = st.selectbox(
        "引擎运行环境",
        engine_keys,
        index=current_engine_idx,
        format_func=lambda k: engine_type_names[k],
        label_visibility="collapsed",
        key="selected_engine_mode"
    )
    if selected_engine != app_cfg.get("engine_type"):
        app_cfg["engine_type"] = selected_engine
        save_app_config(app_cfg)

    # 默认本地模型规格预定义
    MODEL_MAP = {
        "⚡ Qwen3-ASR-1.7B (高精度)": "qwen3_asr_17b",
        "🚀 Qwen3-ASR-0.6B (极速版)": "qwen3_asr_06b"
    }
    selected_label = list(MODEL_MAP.keys())[0]
    selected_res_key = MODEL_MAP[selected_label]
    selected_res = DOWNLOADABLE_RESOURCES[selected_res_key]
    current_model_id = selected_res["ms_model_id"]
    is_sel_ready = True

    if selected_engine == "local":
        selected_label = st.selectbox("核心模型规格", list(MODEL_MAP.keys()), index=0, label_visibility="collapsed")
        selected_res_key = MODEL_MAP[selected_label]
        selected_res = DOWNLOADABLE_RESOURCES[selected_res_key]
        current_model_id = selected_res["ms_model_id"]
        
        sel_path, sel_size = find_local_resource_dir(selected_res)
        is_sel_ready = bool(sel_path and sel_size > 0)
        
        if is_sel_ready:
            st.markdown(f"""
                <div style="padding: 6px 10px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px; margin: 4px 0 8px 0;">
                    <span style="font-size: 0.8rem; color: #10b981; font-weight: 600;">🟢 本地已就绪 ({format_file_size(sel_size)}) • 可离线转写</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="padding: 6px 10px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px; margin: 4px 0 8px 0;">
                    <span style="font-size: 0.8rem; color: #f59e0b; font-weight: 600;">🟡 本地未就绪 ({selected_res['size_estimate']}) • 建议先下载</span>
                </div>
            """, unsafe_allow_html=True)

        total_model_bytes = get_dir_total_size(APP_MODELS_DIR)
        st.markdown(f"""
            <div style="font-size: 0.76rem; opacity: 0.88; margin: 6px 0 8px 0; padding: 6px 8px; background: rgba(255,255,255,0.04); border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
                <div>📁 <b>标准存储</b>: <code>~/Library/Application Support/CutPilot</code></div>
                <div style="margin-top: 3px;">💾 <b>模型总占用</b>: <span style="color:#60a5fa; font-weight:700;">{format_file_size(total_model_bytes)}</span></div>
            </div>
        """, unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("📂 访达直达", key="btn_open_finder_models", use_container_width=True, help="在 macOS 访达中打开标准模型存储目录"):
                subprocess.run(["open", APP_MODELS_DIR])
        with col_m2:
            if st.button("🗑️ 清空模型", key="btn_clean_models", use_container_width=True, help="彻底清空本地已下载的大模型，释放磁盘空间"):
                if os.path.exists(APP_MODELS_DIR):
                    shutil.rmtree(APP_MODELS_DIR, ignore_errors=True)
                    os.makedirs(APP_MODELS_DIR, exist_ok=True)
                    st.toast("✅ 已清空本地模型存储区，磁盘空间已彻底释放！")
                    st.rerun()
    elif selected_engine == "dashscope":
        st.markdown("""
            <div style="padding: 8px 12px; background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; margin: 4px 0 8px 0;">
                <div style="font-size: 0.82rem; color: #60a5fa; font-weight: 600;">☁️ 阿里百炼 (DashScope) 语音识别</div>
                <div style="font-size: 0.73rem; color: var(--text-subtitle); margin-top: 2px;">SenseVoice / Paraformer 极速多语种与分句时码提取</div>
            </div>
        """, unsafe_allow_html=True)
        ds_key_input = st.text_input(
            "DashScope API Key",
            value=app_cfg.get("dashscope_api_key", ""),
            type="password",
            placeholder="sk-...",
            help="阿里云百炼平台 API Key"
        )
        if ds_key_input != app_cfg.get("dashscope_api_key"):
            app_cfg["dashscope_api_key"] = ds_key_input
            save_app_config(app_cfg)
            
        dashscope_model_map = {
            "paraformer-v2": "Paraformer-v2 (影视会议专业版，推荐)",
            "qwen3-asr-flash-filetrans": "Qwen3-ASR-Flash (千问语音大模型)",
            "paraformer-v1": "Paraformer-v1 (经典稳定版)"
        }
        ds_cur_model = app_cfg.get("dashscope_model", "paraformer-v2")
        if ds_cur_model not in dashscope_model_map:
            ds_cur_model = "paraformer-v2"
        ds_model_sel = st.selectbox(
            "百炼识别模型",
            list(dashscope_model_map.keys()),
            index=list(dashscope_model_map.keys()).index(ds_cur_model),
            format_func=lambda k: dashscope_model_map[k]
        )
        if ds_model_sel != app_cfg.get("dashscope_model"):
            app_cfg["dashscope_model"] = ds_model_sel
            save_app_config(app_cfg)
            
        col_ds1, col_ds2 = st.columns(2)
        with col_ds1:
            if st.button("💾 保存配置", key="btn_save_ds", use_container_width=True):
                save_app_config(app_cfg)
                st.success("已保存！")
        with col_ds2:
            if st.button("⚡ 测试连通", key="btn_test_ds", use_container_width=True):
                if not ds_key_input:
                    st.warning("请先输入 API Key！")
                else:
                    with st.spinner("连通性验证中..."):
                        ok, msg = test_dashscope_key(ds_key_input)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
    elif selected_engine == "volcengine":
        st.markdown("""
            <div style="padding: 8px 12px; background: rgba(249, 115, 22, 0.12); border: 1px solid rgba(249, 115, 22, 0.3); border-radius: 8px; margin: 4px 0 8px 0;">
                <div style="font-size: 0.82rem; color: #fb923c; font-weight: 600;">🌋 火山引擎豆包语音识别</div>
                <div style="font-size: 0.73rem; color: var(--text-subtitle); margin-top: 2px;">字节跳动语音大模型，极速流式转写与智能断句</div>
            </div>
        """, unsafe_allow_html=True)
        volc_key_input = st.text_input(
            "火山引擎 API Key / Access Token",
            value=app_cfg.get("volc_api_key", ""),
            type="password",
            placeholder="输入 API Key 或 Token",
            help="火山引擎控制台语音识别授权凭证"
        )
        if volc_key_input != app_cfg.get("volc_api_key"):
            app_cfg["volc_api_key"] = volc_key_input
            save_app_config(app_cfg)
            
        volc_appid_input = st.text_input(
            "App ID (应用标识)",
            value=app_cfg.get("volc_app_id", ""),
            placeholder="例如：1234567890 (可选)",
            help="火山引擎语音应用 AppID"
        )
        if volc_appid_input != app_cfg.get("volc_app_id"):
            app_cfg["volc_app_id"] = volc_appid_input
            save_app_config(app_cfg)
            
        volc_model_map = {
            "volc.bigasr.auc_turbo": "大模型极速版 (Flash API 推荐)",
            "volc.seedasr.auc": "豆包语音识别模型 2.0 (标准版)"
        }
        volc_cur_res = app_cfg.get("volc_resource_id", "volc.bigasr.auc_turbo")
        if volc_cur_res not in volc_model_map:
            volc_cur_res = "volc.bigasr.auc_turbo"
        volc_res_sel = st.selectbox(
            "模型资源 ID",
            list(volc_model_map.keys()),
            index=list(volc_model_map.keys()).index(volc_cur_res),
            format_func=lambda k: volc_model_map[k]
        )
        if volc_res_sel != app_cfg.get("volc_resource_id"):
            app_cfg["volc_resource_id"] = volc_res_sel
            save_app_config(app_cfg)
            
        col_vc1, col_vc2 = st.columns(2)
        with col_vc1:
            if st.button("💾 保存配置", key="btn_save_vc", use_container_width=True):
                save_app_config(app_cfg)
                st.success("已保存！")
        with col_vc2:
            if st.button("⚡ 测试连通", key="btn_test_vc", use_container_width=True):
                if not volc_key_input:
                    st.warning("请先输入 API Key！")
                else:
                    with st.spinner("连通性验证中..."):
                        ok, msg = test_volcengine_key(volc_key_input, volc_appid_input, volc_res_sel)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

    # 折叠面板：本地模型与多源下载管理中心 (仅在选择本地引擎时展开/可用)
    with st.expander("📦 本地模型与多源下载管理", expanded=(selected_engine == "local" and not is_sel_ready)):
        if "mirror_speeds" not in st.session_state:
            try:
                st.session_state["mirror_speeds"] = get_sorted_mirror_speed_results()
            except Exception:
                st.session_state["mirror_speeds"] = [{"id": "modelscope", "name": "阿里魔搭 ModelScope", "latency": 35, "status": "online"}]
        
        cached_speeds = st.session_state.get("mirror_speeds", [])
        fastest_node = cached_speeds[0] if cached_speeds else DOWNLOAD_MIRRORS[0]
        
        st.markdown("<p style='font-size:0.85rem; font-weight:700; margin-bottom:4px;'>🌐 下载服务器优选</p>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            fastest_name_short = fastest_node['name'].split('(')[0].strip()
            mirror_opts = {
                "auto": f"⚡ 自动优选 ({fastest_name_short} {fastest_node['latency']}ms)",
                "modelscope": "阿里魔搭 (国内高速 CDN)",
                "hf_mirror": "HF-Mirror (国内镜像站)",
                "huggingface": "Hugging Face (官方海外源)"
            }
            chosen_mirror_key = st.selectbox(
                "下载镜像线路", 
                list(mirror_opts.keys()), 
                format_func=lambda k: mirror_opts[k], 
                label_visibility="collapsed",
                key="mirror_choice_select"
            )
        with col_m2:
            if st.button("🔄 测速", use_container_width=True, help="立即并发测试三大镜像源最新延迟"):
                with st.spinner("测速中..."):
                    st.session_state["mirror_speeds"] = get_sorted_mirror_speed_results()
                    st.rerun()

        st.markdown("<div style='height: 1px; background: var(--border-card); margin: 6px 0 10px 0;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; font-weight:700; margin-bottom:6px;'>📑 模型与资源自选列表</p>", unsafe_allow_html=True)

        for r_key, r_info in DOWNLOADABLE_RESOURCES.items():
            r_p, r_sz = find_local_resource_dir(r_info)
            r_ok = bool(r_p and r_sz > 0)
            
            with st.container():
                st.markdown(f"<span style='font-size:0.88rem; font-weight:700;'>{r_info['title']}</span>", unsafe_allow_html=True)
                st.caption(f"{r_info['desc']} • 体积: {r_info['size_estimate']}")
                
                col_st, col_bt = st.columns([2, 1])
                with col_st:
                    if r_ok:
                        st.markdown(f"<span style='color:#10b981; font-size:0.8rem; font-weight:600;'>✅ 已下载 ({format_file_size(r_sz)})</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#f59e0b; font-size:0.8rem; font-weight:600;'>⚠️ 未就绪</span>", unsafe_allow_html=True)
                with col_bt:
                    btn_text = "重新下载" if r_ok else "📥 下载"
                    if st.button(btn_text, key=f"btn_dl_{r_key}", use_container_width=True):
                        try:
                            active_node = fastest_node if chosen_mirror_key == "auto" else next((m for m in DOWNLOAD_MIRRORS if m["id"] == chosen_mirror_key), fastest_node)
                            dl_progress_bar = st.progress(0.0, text="0%")
                            capture = TqdmProgressCapture(dl_progress_bar)
                            sys.stderr = capture
                            try:
                                execute_resource_download(r_key, chosen_mirror_key)
                            finally:
                                sys.stderr = capture.original_stderr
                            
                            dl_progress_bar.empty()
                            st.success(f"🎉 {r_info['title']} 下载完成并就绪！")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as dl_err:
                            st.error(f"下载失败: {dl_err}")
                st.markdown("<div style='height: 1px; background: var(--border-card); margin: 4px 0 8px 0; opacity: 0.5;'></div>", unsafe_allow_html=True)
    st.caption("💡 引擎提示：优先启用 `qwen_asr` 原生驱动；若未安装将自动使用 `transformers` 适配层。推荐终端执行: `pip install qwen-asr soundfile`")

    st.markdown("<br><p style='font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;'>📏 视觉排版规范</p>", unsafe_allow_html=True)
    target_max_chars = st.slider(
        "单行字幕安全字数", 
        min_value=8, max_value=22, value=14, step=1,
        help="推荐 14 字，符合主流影视后期与短视频字幕安全区标准"
    )

    st.markdown("<br><p style='font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;'>🎬 工程导出参数</p>", unsafe_allow_html=True)
    fps = st.selectbox("时间线帧率 (FPS)", [25, 24, 23.976, 29.97, 30, 60], index=0)
    
    st.markdown("""
        <div style="margin-top: 40px; padding: 14px; background: var(--badge-bg); border: 1px solid var(--badge-border); border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">🔒</span>
                <div>
                    <div style="color: var(--badge-text-title); font-weight: 700; font-size: 0.9rem;">本地转写与处理</div>
                    <div style="color: var(--badge-text-sub); font-size: 0.75rem; margin-top: 2px;">云端审校仅在手动启用后发送台词与时码</div>
                </div>
            </div>
        </div>
        
        <div style="position: fixed; bottom: 12px; left: 18px; z-index: 9999; pointer-events: none; user-select: none;">
            <span style="font-size: 0.72rem; color: var(--text-subtitle); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; letter-spacing: 0.6px; opacity: 0.85;">
                JChen_v1.6
            </span>
        </div>
    """, unsafe_allow_html=True)

# ----------------- 注入主题与高级定制 CSS -----------------
if "浅色" in theme_choice:
    theme_vars = """
        --bg-main: #f8fafc;
        --bg-sidebar: #f1f5f9;
        --bg-card: #ffffff;
        --border-card: #e2e8f0;
        --text-title: #0f172a;
        --text-subtitle: #64748b;
        --text-body: #1e293b;
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
        --input-text: #0f172a;
        --card-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.06);
        --card-hover-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.12);
        --primary-accent: #2563eb;
        --uploader-bg: #ffffff;
        --tab-bg: #f1f5f9;
        --badge-bg: rgba(16, 185, 129, 0.08);
        --badge-border: rgba(16, 185, 129, 0.25);
        --badge-text-title: #047857;
        --badge-text-sub: #475569;
    """
    sys_override = ""
elif "深色" in theme_choice:
    theme_vars = """
        --bg-main: #0b0f19;
        --bg-sidebar: #070a13;
        --bg-card: #141c2e;
        --border-card: #283548;
        --text-title: #f8fafc;
        --text-subtitle: #94a3b8;
        --text-body: #cbd5e1;
        --input-bg: #1a2436;
        --input-border: #334155;
        --input-text: #f8fafc;
        --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        --card-hover-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.7);
        --primary-accent: #3b82f6;
        --uploader-bg: #141c2e;
        --tab-bg: #0f172a;
        --badge-bg: rgba(16, 185, 129, 0.12);
        --badge-border: rgba(16, 185, 129, 0.25);
        --badge-text-title: #10b981;
        --badge-text-sub: #94a3b8;
    """
    sys_override = ""
else: # 跟随系统外观
    theme_vars = """
        --bg-main: #0b0f19;
        --bg-sidebar: #070a13;
        --bg-card: #141c2e;
        --border-card: #283548;
        --text-title: #f8fafc;
        --text-subtitle: #94a3b8;
        --text-body: #cbd5e1;
        --input-bg: #1a2436;
        --input-border: #334155;
        --input-text: #f8fafc;
        --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        --card-hover-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.7);
        --primary-accent: #3b82f6;
        --uploader-bg: #141c2e;
        --tab-bg: #0f172a;
        --badge-bg: rgba(16, 185, 129, 0.12);
        --badge-border: rgba(16, 185, 129, 0.25);
        --badge-text-title: #10b981;
        --badge-text-sub: #94a3b8;
    """
    sys_override = """
        @media (prefers-color-scheme: light) {
            :root, .stApp {
                --bg-main: #f8fafc !important;
                --bg-sidebar: #f1f5f9 !important;
                --bg-card: #ffffff !important;
                --border-card: #e2e8f0 !important;
                --text-title: #0f172a !important;
                --text-subtitle: #64748b !important;
                --text-body: #1e293b !important;
                --input-bg: #ffffff !important;
                --input-border: #cbd5e1 !important;
                --input-text: #0f172a !important;
                --card-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.06) !important;
                --card-hover-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.12) !important;
                --primary-accent: #2563eb !important;
                --uploader-bg: #ffffff !important;
                --tab-bg: #f1f5f9 !important;
                --badge-bg: rgba(16, 185, 129, 0.08) !important;
                --badge-border: rgba(16, 185, 129, 0.25) !important;
                --badge-text-title: #047857 !important;
                --badge-text-sub: #475569 !important;
            }
        }
    """

st.markdown(f"""
    <style>
        :root {{
            {theme_vars}
        }}
        {sys_override}

        /* 隐藏原生的 header 和 footer */
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* 隐藏侧边栏收起按钮及 X 关闭按钮 */
        [data-testid="collapsedControl"] {{ display: none !important; }}
        section[data-testid="stSidebar"] button[kind="header"] {{ display: none !important; }}
        
        /* 全局字体与排版优化 */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], section.main {{
            background-color: var(--bg-main) !important;
            color: var(--text-body) !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        /* 侧边栏整体美化 */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-card) !important;
        }}
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {{
            color: var(--text-title);
        }}
        [data-testid="stSidebar"] .caption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            color: var(--text-subtitle) !important;
        }}

        /* 顶部 Logo 栏区域 */
        .brand-header {{
            display: flex;
            align-items: center;
            padding: 0.5rem 0 1.5rem 0;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 1.8rem;
        }}
        .brand-text h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--text-title) !important;
        }}
        .brand-text p {{
            margin: 4px 0 0 0;
            font-size: 1rem;
            color: var(--text-subtitle) !important;
            font-weight: 500;
        }}

        /* 卡片容器：针对 st.container(border=True) 赋予现代立体卡片质感 */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: var(--bg-card) !important;
            border-radius: 16px !important;
            border: 1px solid var(--border-card) !important;
            box-shadow: var(--card-shadow) !important;
            padding: 22px 24px !important;
            margin-bottom: 24px !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: var(--primary-accent) !important;
            box-shadow: var(--card-hover-shadow) !important;
        }}

        /* 标题带炫彩左边框 */
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            color: var(--text-title) !important;
            display: flex;
            align-items: center;
        }}
        .section-title::before {{
            content: '';
            display: inline-block;
            width: 5px;
            height: 22px;
            background: linear-gradient(180deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 4px;
            margin-right: 12px;
        }}

        /* 按钮美化 */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            font-size: 15px;
            letter-spacing: 0.5px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
        }}
        .stButton > button[kind="secondary"] {{
            background: var(--input-bg) !important;
            color: var(--text-title) !important;
            border: 1px solid var(--input-border) !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            border-color: var(--primary-accent) !important;
            color: var(--primary-accent) !important;
            transform: translateY(-2px) !important;
        }}

        /* 标签页优化 */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: var(--tab-bg) !important;
            padding: 0.5rem 1rem;
            border-radius: 12px 12px 0 0;
            border-bottom: 1px solid var(--border-card);
            gap: 20px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: var(--text-subtitle) !important;
            font-weight: 600;
            font-size: 0.95rem;
            padding-bottom: 10px;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--primary-accent) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: var(--primary-accent) !important;
            height: 3px;
            border-radius: 3px 3px 0 0;
        }}

        /* 输入框和 SelectBox */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {{
            background-color: var(--input-bg) !important;
            border-radius: 8px !important;
            border: 1px solid var(--input-border) !important;
            color: var(--input-text) !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="input"] input {{
            color: var(--input-text) !important;
        }}

        /* 文件上传区域美化 */
        [data-testid="stFileUploader"] section {{
            background-color: var(--uploader-bg) !important;
            border: 1.5px dashed var(--border-card) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stFileUploader"] section:hover {{
            border-color: var(--primary-accent) !important;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small {{
            color: var(--text-subtitle) !important;
        }}

        /* 数据表格与编辑器 */
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            background-color: var(--bg-card) !important;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-card);
        }}

        /* 全局标题与文本色彩确保高对比度 */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-title) !important;
        }}
        p, span, label, div[data-testid="stMarkdownContainer"] p {{
            color: var(--text-body);
        }}
    </style>
""", unsafe_allow_html=True)

# ----------------- 本地文件保存弹窗 -----------------
def save_file_locally(content, default_name, ext, is_bytes=False):
    """
    通过 macOS 原生 AppleScript 唤起系统级置顶保存对话框。
    彻底摆脱对外部 Python 运行时与 AppKit/Tkinter 依赖，100% 解决窗口压底与无响应问题。
    """
    clean_ext = ext.lstrip(".")
    full_default = default_name if default_name.endswith(ext) else f"{default_name}{ext}"

    # 1. 严格转义字符：反斜杠和双引号
    escaped_default = full_default.replace('\\', '\\\\').replace('"', '\\"')

    # 2. 标准 AppleScript 脚本：仅对 System Events 设 4 秒探测超时，用户选文件对话框不设超时限制
    osa_code = f'''
    set frontAppName to "Finder"
    try
        with timeout of 4 seconds
            tell application "System Events"
                set frontAppName to name of first process whose frontmost is true
            end tell
        end timeout
    end try
    try
        tell application frontAppName
            activate
            set chosenFile to choose file name with prompt "请选择文件保存路径：" default name "{escaped_default}"
            return POSIX path of chosenFile
        end tell
    on error errMsg number errNum
        if errNum is -128 then
            return "CANCELED"
        end if
        return "ERROR:" & errMsg
    end try
    '''

    fallback_osa = f'''
    try
        set f to choose file name with prompt "请选择文件保存路径：" default name "{escaped_default}"
        return POSIX path of f
    on error errMsg number errNum
        if errNum is -128 then
            return "CANCELED"
        end if
        return "ERROR:" & errMsg
    end try
    '''

    try:
        res = subprocess.run(["osascript", "-e", osa_code], capture_output=True, text=True, check=False)
        output = res.stdout.strip()
        if output == "CANCELED":
            st.info("已取消保存操作")
            return False
        elif output.startswith("ERROR:") or res.returncode != 0:
            # 回退至无进程绑定的基础 osascript 方式
            fb_res = subprocess.run(["osascript", "-e", fallback_osa], capture_output=True, text=True, check=False)
            output = fb_res.stdout.strip()
            if output == "CANCELED":
                st.info("已取消保存操作")
                return False
            elif fb_res.returncode != 0 or not output or output.startswith("ERROR:"):
                st.info("已取消保存或未选择路径")
                return False

        path = output
        if path:
            if not path.endswith(ext):
                path = f"{path}{ext}"
            mode = "wb" if is_bytes else "w"
            encoding = None if is_bytes else "utf-8"
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            st.success(f"✅ 已成功保存至: {path}")
            return True
        return False
    except Exception as e:
        # 终极保底：保存至当前用户的 Downloads 目录，遇同名自动递增序号防止覆盖
        try:
            stem, suffix = os.path.splitext(full_default)
            downloads_dir = os.path.expanduser("~/Downloads")
            fallback_path = os.path.join(downloads_dir, full_default)
            counter = 1
            while os.path.exists(fallback_path):
                fallback_path = os.path.join(downloads_dir, f"{stem}_{counter}{suffix}")
                counter += 1
            mode = "wb" if is_bytes else "w"
            encoding = None if is_bytes else "utf-8"
            os.makedirs(os.path.dirname(os.path.abspath(fallback_path)), exist_ok=True)
            with open(fallback_path, mode, encoding=encoding) as f:
                f.write(content)
            st.success(f"✅ 已备用保存至下载目录: {fallback_path}")
            return True
        except Exception as fb_err:
            st.error(f"保存失败: {fb_err}")
            return False

def cleanup_session_temp_files():
    for path in st.session_state.pop("_temp_files", []):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

def track_temp_file(path):
    st.session_state.setdefault("_temp_files", []).append(path)
    return path

# ----------------- 核心音视频处理函数 -----------------
def detect_media_streams(media_path):
    """检测素材是否包含视频流与音频流。"""
    ext = Path(media_path).suffix.lower()
    audio_only_exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
    if ext in audio_only_exts:
        return {"has_video": False, "has_audio": True}
    
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        has_video = "video" in result.stdout.lower()
    except Exception:
        has_video = ext in {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
    return {"has_video": has_video, "has_audio": True}

def extract_audio(input_media_path, output_wav_path):
    cmd = ["ffmpeg", "-y", "-i", input_media_path, "-ac", "1", "-ar", "16000", "-vn", output_wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def generate_preview_audio(wav_path):
    """为前端流式播放生成高压缩比的轻量 MP3 预览音频，极大节省内存与传输带宽。"""
    if not wav_path or not os.path.exists(wav_path):
        return wav_path
    mp3_path = os.path.splitext(wav_path)[0] + "_preview.mp3"
    if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
        return mp3_path
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ac", "1",
        "-ar", "16000",
        "-codec:a", "libmp3lame",
        "-b:a", "48k",
        mp3_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return track_temp_file(mp3_path)
    except Exception:
        return wav_path

def get_streamable_media_url(file_path, mimetype="audio/mp3"):
    """尝试通过 Streamlit 的内部 MediaFileManager 将本地音频注册为流式 HTTP 端点。"""
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        from streamlit.runtime import exists, get_instance
        if exists():
            return get_instance().media_file_mgr.add(file_path, mimetype, "cutpilot_continuous_player")
    except Exception:
        pass
    return ""

def get_audio_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def get_fps_spec(fps_val):
    """返回实际帧率的分子/分母及显示用名义帧率。"""
    if math.isclose(fps_val, 23.976, abs_tol=0.001):
        return 24000, 1001, 24
    if math.isclose(fps_val, 29.97, abs_tol=0.001):
        return 30000, 1001, 30
    nominal = int(round(fps_val))
    return nominal, 1, nominal

def frames_to_fcpx_time(frames, fps_val):
    rate_num, rate_den, _ = get_fps_spec(fps_val)
    return f"{frames * rate_den}/{rate_num}s"

def format_timecode(seconds, fps_val):
    rate_num, rate_den, nominal_fps = get_fps_spec(fps_val)
    total_frames = max(0, int(round(seconds * rate_num / rate_den)))
    ff = total_frames % nominal_fps
    total_seconds = total_frames // nominal_fps
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

def format_timecode_ms(seconds, separator=','):
    total_ms = max(0, int(round(seconds * 1000)))
    hh, remainder = divmod(total_ms, 3_600_000)
    mm, remainder = divmod(remainder, 60_000)
    ss, ms = divmod(remainder, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{separator}{ms:03d}"

# ----------------- 语音转写推理 -----------------
class TransformerASRAdapter:
    """当未安装 qwen_asr 时，使用 transformers AutoProcessor + AutoModel 进行特征提取与解码的兜底适配器"""
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def transcribe(self, audio, language="Chinese"):
        import soundfile as sf
        import torch
        wav, sr = sf.read(audio)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        if hasattr(self.processor, "__call__"):
            inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(self.model, "generate"):
                    generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                    if hasattr(self.processor, "batch_decode"):
                        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        transcription = str(generated_ids)
                else:
                    raise RuntimeError("当前模型不支持 generate 方法")
            return transcription.strip()
        else:
            raise RuntimeError("AutoProcessor 初始化异常，无法提取音频特征")

@st.cache_resource(show_spinner=False)
def load_local_model(model_id):
    import torch
    # 优先检测本地已存在的目录，若已就绪直接载入，免联网检查
    found_dir = None
    for res in DOWNLOADABLE_RESOURCES.values():
        if res.get("ms_model_id") == model_id or res.get("hf_model_id") == model_id:
            p, sz = find_local_resource_dir(res)
            if p and sz > 0:
                found_dir = p
                break
    
    if found_dir:
        model_dir = found_dir
    else:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 方案 1: 优先尝试官方原生 qwen_asr
    try:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map=device
        )
        return model
    except ImportError:
        pass
    except Exception as e:
        print(f"通过 qwen_asr 加载模型失败 ({e})，尝试使用 transformers 适配层...")

    # 方案 2: 使用 transformers AutoProcessor + AutoModel / AutoModelForCausalLM 适配
    try:
        from transformers import AutoProcessor, AutoModel
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
            ).to(device)
        except Exception:
            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
            ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        return TransformerASRAdapter(model, processor, device)
    except Exception as exc:
        raise RuntimeError(
            f"加载语音识别引擎失败！\n"
            f"建议在终端执行安装推荐依赖包：\n"
            f"  pip install qwen-asr soundfile\n"
            f"详细错误: {exc}"
        ) from exc

def run_base_transcribe(model, wav_path, total_dur):
    chunk_len = 15.0
    curr = 0.0
    raw_segments = []
    total_steps = max(1, int((total_dur + chunk_len - 1) // chunk_len))
    prog = st.progress(0.0)
    step = 0
    
    while curr < total_dur:
        dur = min(chunk_len, total_dur - curr)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_c:
            c_path = tmp_c.name
        cmd = ["ffmpeg", "-y", "-ss", str(curr), "-i", wav_path, "-t", str(dur), "-c", "copy", c_path]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if hasattr(model, "transcribe"):
                r = model.transcribe(audio=c_path, language="Chinese")
                text = r[0].text if isinstance(r, list) else (r if isinstance(r, str) else getattr(r, "text", str(r)))
            elif hasattr(model, "generate"):
                text = str(model.generate(c_path))
            else:
                raise RuntimeError("模型对象缺少有效的 transcribe 或 generate 接口")
        except Exception as exc:
            raise RuntimeError(f"{curr:.2f}s–{curr + dur:.2f}s 的转写失败：{exc}") from exc
        finally:
            if os.path.exists(c_path):
                os.remove(c_path)
            
        text = text.strip()
        if text:
            raw_segments.append({
                "保留": True,
                "开始(s)": round(curr, 2),
                "结束(s)": round(curr + dur, 2),
                "标签": "#普通",
                "台词内容": text
            })
        curr += dur
        step += 1
        prog.progress(min(1.0, step / total_steps))
    prog.empty()
    return raw_segments

# ----------------- 阿里百炼与火山引擎云端语音引擎 -----------------
def transcribe_dashscope(api_key, wav_path, total_dur, model="paraformer-v2"):
    import requests
    import time
    if not api_key or not api_key.strip():
        raise ValueError("请先在左侧边栏填写有效的阿里百炼 API Key！")
    
    api_key = api_key.strip()
    headers_auth = {"Authorization": f"Bearer {api_key}"}
    
    # 步骤 1: 上传本地音频到百炼临时推理存储
    upload_url = "https://dashscope.aliyuncs.com/api/v1/files"
    filename = os.path.basename(wav_path)
    with open(wav_path, "rb") as f:
        files = {"file": (filename, f, "audio/wav")}
        data = {"purpose": "inference"}
        res = requests.post(upload_url, headers=headers_auth, files=files, data=data, timeout=60)
    if res.status_code != 200:
        err_msg = res.text
        try:
            err_msg = res.json().get("message", res.text)
        except Exception:
            pass
        raise RuntimeError(f"百炼音频上传失败 (HTTP {res.status_code}): {err_msg}")
        
    upload_res = res.json()
    file_id = upload_res.get("output", {}).get("uploaded_files", [{}])[0].get("file_id")
    if not file_id:
        raise RuntimeError(f"解析百炼上传凭证失败: {upload_res}")
        
    file_info_url = f"https://dashscope.aliyuncs.com/api/v1/files/{file_id}"
    res_info = requests.get(file_info_url, headers=headers_auth, timeout=30)
    if res_info.status_code != 200:
        raise RuntimeError(f"获取百炼文件详情失败: {res_info.text}")
    audio_url = res_info.json().get("output", {}).get("url")
    if not audio_url:
        raise RuntimeError(f"未能获取音频访问地址: {res_info.text}")
        
    # 步骤 2: 提交异步转写任务
    asr_submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    payload = {
        "model": model,
        "input": {"file_urls": [audio_url]},
        "parameters": {"enable_words": True, "enable_itn": True}
    }
    submit_res = requests.post(asr_submit_url, headers={**headers_auth, "Content-Type": "application/json"}, json=payload, timeout=30)
    if submit_res.status_code != 200:
        raise RuntimeError(f"提交百炼语音转写任务失败: {submit_res.text}")
    task_id = submit_res.json().get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"未能获取转写任务 ID: {submit_res.json()}")
        
    # 步骤 3: 轮询结果
    query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    max_wait = 360
    start_t = time.time()
    raw_segments = []
    
    status_placeholder = st.empty()
    while time.time() - start_t < max_wait:
        poll_res = requests.get(query_url, headers=headers_auth, timeout=30)
        if poll_res.status_code == 200:
            poll_json = poll_res.json()
            status = poll_json.get("output", {}).get("task_status")
            status_placeholder.info(f"⏳ 阿里百炼转写处理中 (状态: {status}, 已耗时 {int(time.time() - start_t)}s)...")
            if status == "SUCCEEDED":
                status_placeholder.empty()
                results = poll_json.get("output", {}).get("results", [])
                if results and "transcription_url" in results[0]:
                    trans_url = results[0]["transcription_url"]
                    content_res = requests.get(trans_url, timeout=60)
                    detail_data = content_res.json()
                    for tr in detail_data.get("transcripts", []):
                        sentences = tr.get("sentences", [])
                        if sentences:
                            for s in sentences:
                                text = s.get("text", "").strip()
                                if text:
                                    raw_segments.append({
                                        "保留": True,
                                        "开始(s)": round(s.get("begin_time", 0) / 1000.0, 2),
                                        "结束(s)": round(s.get("end_time", 0) / 1000.0, 2),
                                        "标签": "#普通",
                                        "台词内容": text
                                    })
                        elif tr.get("text"):
                            raw_segments.append({
                                "保留": True,
                                "开始(s)": 0.0,
                                "结束(s)": round(total_dur, 2),
                                "标签": "#普通",
                                "台词内容": tr.get("text").strip()
                            })
                break
            elif status in ["FAILED", "CANCELED"]:
                status_placeholder.empty()
                code = poll_json.get("output", {}).get("code", "")
                msg = poll_json.get("output", {}).get("message", "任务转写失败")
                raise RuntimeError(f"百炼转写失败 ({code}): {msg}")
        time.sleep(2)
        
    status_placeholder.empty()
    if not raw_segments:
        if time.time() - start_t >= max_wait:
            raise TimeoutError("阿里百炼语音识别超时（已等待 6 分钟），请重试或检查音频文件。")
    return raw_segments

def transcribe_volcengine(api_key, app_id, resource_id, wav_path, total_dur):
    import requests
    import uuid
    import base64
    if not api_key or not api_key.strip():
        raise ValueError("请先在左侧边栏填写有效的火山引擎 API Key / Access Token！")
        
    api_key = api_key.strip()
    resource_id = (resource_id or "volc.bigasr.auc_turbo").strip()
    flash_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    
    chunk_len = 50.0
    curr = 0.0
    raw_segments = []
    total_steps = max(1, int((total_dur + chunk_len - 1) // chunk_len))
    prog = st.progress(0.0, text="正在通过火山引擎豆包大模型识别...")
    step = 0
    
    while curr < total_dur:
        dur = min(chunk_len, total_dur - curr)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_c:
            c_path = tmp_c.name
            
        cmd = ["ffmpeg", "-y", "-ss", str(curr), "-i", wav_path, "-t", str(dur), "-c", "copy", c_path]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with open(c_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
                
            headers = {
                "X-Api-Key": api_key,
                "X-Api-Resource-Id": resource_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
                "Content-Type": "application/json"
            }
            if app_id and app_id.strip():
                headers["X-Api-App-Key"] = app_id.strip()
            if not api_key.startswith("volc_") and len(api_key) > 20:
                headers["Authorization"] = f"Bearer; {api_key}"
                
            payload = {
                "user": {"uid": f"cutpilot_{uuid.uuid4().hex[:8]}"},
                "audio": {"format": "wav", "data": b64_data},
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "show_utterances": True
                }
            }
            res = requests.post(flash_url, headers=headers, json=payload, timeout=60)
            status_code_header = res.headers.get("X-Api-Status-Code")
            if status_code_header and status_code_header != "20000000":
                msg = res.headers.get("X-Api-Message", res.text)
                raise RuntimeError(f"火山引擎返回异常 ({status_code_header}): {msg}")
                
            res_json = res.json()
            result = res_json.get("result", {})
            utterances = result.get("utterances", [])
            if utterances:
                for u in utterances:
                    txt = u.get("text", "").strip()
                    if txt:
                        s_sec = round(curr + (u.get("start_time", 0) / 1000.0), 2)
                        e_sec = round(curr + (u.get("end_time", dur * 1000) / 1000.0), 2)
                        raw_segments.append({
                            "保留": True,
                            "开始(s)": s_sec,
                            "结束(s)": e_sec,
                            "标签": "#普通",
                            "台词内容": txt
                        })
            else:
                chunk_txt = result.get("text", "").strip()
                if chunk_txt:
                    raw_segments.append({
                        "保留": True,
                        "开始(s)": round(curr, 2),
                        "结束(s)": round(curr + dur, 2),
                        "标签": "#普通",
                        "台词内容": chunk_txt
                    })
        finally:
            if os.path.exists(c_path):
                os.remove(c_path)
                
        curr += dur
        step += 1
        prog.progress(min(1.0, step / total_steps), text=f"火山引擎识别进度: {step}/{total_steps}")
        
    prog.empty()
    return raw_segments

# ----------------- 100% 逐字无损规整算法 -----------------
def lossless_resegment_to_target(df, max_chars=14):
    char_stream = []
    for _, row in df.iterrows():
        text = str(row.get("台词内容", ""))
        s = float(row.get("开始(s)", 0.0))
        e = float(row.get("结束(s)", 0.0))
        tag = row.get("标签", "#普通")
        keep = row.get("保留", True)
        
        if not text:
            continue
            
        dur = max(0.05, e - s)
        n = len(text)
        char_dur = dur / n
        
        for idx, ch in enumerate(text):
            ch_s = s + idx * char_dur
            ch_e = ch_s + char_dur
            char_stream.append({
                "char": ch,
                "s": ch_s,
                "e": ch_e,
                "tag": tag,
                "keep": keep
            })
            
    if not char_stream:
        return df

    new_rows = []
    i = 0
    total_len = len(char_stream)
    
    while i < total_len:
        window_end = min(total_len, i + max_chars)
        window_chars = [char_stream[k]["char"] for k in range(i, window_end)]
        
        split_at = len(window_chars)
        # 不跨越已有的保留状态或标签边界，避免重排后改变剪辑决策。
        first = char_stream[i]
        for local_idx, item in enumerate(char_stream[i + 1:window_end], start=1):
            if item["tag"] != first["tag"] or item["keep"] != first["keep"]:
                split_at = local_idx
                break
        if window_end < total_len:
            # 1. 优先在常见标点符号处断开
            punct_found = False
            for idx_back in range(split_at - 1, max(3, split_at - 6), -1):
                if window_chars[idx_back] in ["，", ",", "。", "！", "？", "；", ";", "、", " ", "…"]:
                    split_at = idx_back + 1
                    punct_found = True
                    break
            # 2. 词边界保护：避免在英文字词、数字或复合连词中间强行切断
            if not punct_found and split_at > 4 and (i + split_at < total_len):
                prev_c = char_stream[i + split_at - 1]["char"]
                next_c = char_stream[i + split_at]["char"]
                if prev_c.isalnum() and next_c.isalnum():
                    for idx_back in range(split_at - 1, max(3, split_at - 6), -1):
                        if not char_stream[i + idx_back]["char"].isalnum():
                            split_at = idx_back + 1
                            break
                    
        line_chars = char_stream[i : i + split_at]
        line_text = "".join([c["char"] for c in line_chars])
        line_s = round(line_chars[0]["s"], 2)
        line_e = round(line_chars[-1]["e"], 2)
        line_tag = line_chars[0]["tag"]
        line_keep = line_chars[0]["keep"]
        
        if line_text:
            new_rows.append({
                "保留": line_keep,
                "开始(s)": line_s,
                "结束(s)": line_e,
                "标签": line_tag,
                "台词内容": line_text
            })
        i += split_at
        
    res_df = pd.DataFrame(new_rows)
    # SubFix 风格文本守恒断言：确保重排规整前后字符绝对 0 丢失
    orig_text = strip_punctuation_and_spaces("".join(str(r) for r in df["台词内容"]))
    new_text = strip_punctuation_and_spaces("".join(str(r) for r in res_df["台词内容"]))
    if orig_text != new_text:
        return df
    return res_df

# ----------------- AI 语义断句与时码再校准算法 -----------------
def local_semantic_realign(df, max_chars=14, max_gap=1.2):
    raw_list = []
    for _, row in df.iterrows():
        raw_list.append({
            "start": float(row.get("开始(s)", 0.0)),
            "end": float(row.get("结束(s)", 0.0)),
            "tag": str(row.get("标签", "#普通")),
            "text": str(row.get("台词内容", "")).strip(),
            "keep": bool(row.get("保留", True))
        })
    if not raw_list:
        return df

    dangling_ends = ("的", "地", "得", "和", "跟", "与", "同", "及", "而", "因为", "所以", "如果", "虽然", 
                     "主要", "把", "被", "在", "向", "往", "关于", "对于", "这个", "那个", "一种", "一个", "就是", "还有", "当时")
    dangling_starts = ("的", "地", "得", "了", "着", "过", "呢", "吧", "啊", "呀", "吗", "嘛", "啦", "等等", "来说", "而言", "去", "做")

    merged = []
    i = 0
    while i < len(raw_list):
        cur = dict(raw_list[i])
        while i + 1 < len(raw_list):
            next_seg = raw_list[i + 1]
            gap = next_seg["start"] - cur["end"]
            combined_len = len(cur["text"]) + len(next_seg["text"])

            should_merge = False
            if gap <= max_gap * 1.5 and (any(cur["text"].endswith(w) for w in dangling_ends) or any(next_seg["text"].startswith(w) for w in dangling_starts)):
                should_merge = True
            elif gap <= max_gap and (len(cur["text"]) <= 4 or len(next_seg["text"]) <= 4) and combined_len <= max_chars + 4:
                should_merge = True
            elif gap <= 0.35 and combined_len <= max_chars:
                should_merge = True

            if should_merge:
                # 保留/剔除和标签属于人工编辑决策，不能在语义合并时悄悄吞掉。
                if next_seg["keep"] != cur["keep"] or next_seg["tag"] != cur["tag"]:
                    should_merge = False

            if should_merge:
                cur["text"] = cur["text"] + next_seg["text"]
                cur["end"] = next_seg["end"]
                i += 1
            else:
                break
        merged.append(cur)
        i += 1

    final_rows = []
    for item in merged:
        text = item["text"].strip()
        start = item["start"]
        end = item["end"]
        dur = max(0.1, end - start)
        tag = item["tag"]
        keep = item["keep"]

        if len(text) <= max_chars:
            final_rows.append({
                "保留": keep,
                "开始(s)": round(start, 2),
                "结束(s)": round(end, 2),
                "标签": tag,
                "台词内容": text
            })
            continue

        pieces = []
        rem = text
        while len(rem) > max_chars:
            window = rem[:max_chars + 2]
            found = False
            for sep in ["，", ",", " ", "、", "；", ";", "！", "？", "：", "。"]:
                pos = window.rfind(sep)
                if pos >= 4:
                    cut_idx = pos if sep == " " else pos + 1
                    pieces.append(rem[:cut_idx].strip())
                    rem = rem[pos+1:].strip()
                    found = True
                    break
            if not found:
                for sep in ["的", "了", "在", "是", "把"]:
                    pos = window.rfind(sep)
                    if 4 <= pos <= max_chars:
                        pieces.append(rem[:pos+1].strip())
                        rem = rem[pos+1:].strip()
                        found = True
                        break
            if not found:
                pieces.append(rem[:max_chars].strip())
                rem = rem[max_chars:].strip()
        if rem:
            pieces.append(rem.strip())

        if len(pieces) > 1 and len(pieces[-1]) <= 3:
            combined_p = pieces[-2] + pieces[-1]
            if len(combined_p) <= max_chars:
                pieces = pieces[:-2] + [combined_p]
            else:
                mid = len(combined_p) // 2
                pieces[-2] = combined_p[:mid]
                pieces[-1] = combined_p[mid:]

        total_len = sum(len(p) for p in pieces)
        curr_t = start
        for idx_p, p in enumerate(pieces):
            p_dur = round(dur * (len(p) / max(1, total_len)), 2)
            p_end = round(curr_t + p_dur, 2)
            if idx_p == len(pieces) - 1:
                p_end = end
            final_rows.append({
                "保留": keep,
                "开始(s)": round(curr_t, 2),
                "结束(s)": round(p_end, 2),
                "标签": tag,
                "台词内容": p
            })
            curr_t = p_end

    return pd.DataFrame(final_rows)

def strip_punctuation_and_spaces(text):
    """去除中英文标点符号与空白字符，仅保留核心文本用于内容一致性比对。"""
    return re.sub(r"[\s\.,\/#!$%\^&\*;:{}=\-_`~()\"'，。！？、“”‘’：；（）《》、\n\r\t]+", "", str(text))

def split_oversized_segment(text, start, end, tag, max_chars):
    """当 LLM 返回单句超过 max_chars 时，自动在标点、连接词或字数中间做平滑二次切分，平分时码，避免硬报错阻断。"""
    if len(text) <= max_chars:
        return [{"start": round(start, 2), "end": round(end, 2), "tag": tag, "text": text}]

    dur = max(0.1, end - start)
    pieces = []
    rem = text
    while len(rem) > max_chars:
        window = rem[:max_chars + 2]
        found = False
        for sep in ["，", ",", " ", "、", "；", ";", "！", "？", "：", "。"]:
            pos = window.rfind(sep)
            if pos >= 3:
                pieces.append(rem[:pos+1].strip())
                rem = rem[pos+1:].strip()
                found = True
                break
        if not found:
            for sep in ["或者", "通过", "但是", "因为", "所以", "如果", "虽然", "以及", "的", "了", "在", "是", "把"]:
                pos = window.rfind(sep)
                if 3 <= pos <= max_chars:
                    cut_pos = pos + len(sep)
                    pieces.append(rem[:cut_pos].strip())
                    rem = rem[cut_pos:].strip()
                    found = True
                    break
        if not found:
            pieces.append(rem[:max_chars].strip())
            rem = rem[max_chars:].strip()
    if rem:
        pieces.append(rem.strip())

    if len(pieces) > 1 and len(pieces[-1]) <= 2:
        combined = pieces[-2] + pieces[-1]
        if len(combined) <= max_chars:
            pieces = pieces[:-2] + [combined]
        else:
            mid = len(combined) // 2
            pieces[-2] = combined[:mid]
            pieces[-1] = combined[mid:]

    total_chars = sum(len(p) for p in pieces)
    results = []
    curr_t = start
    for idx, p in enumerate(pieces):
        p_dur = round(dur * (len(p) / max(1, total_chars)), 2)
        p_end = round(curr_t + p_dur, 2) if idx < len(pieces) - 1 else end
        results.append({
            "start": round(curr_t, 2),
            "end": round(p_end, 2),
            "tag": tag,
            "text": p
        })
        curr_t = p_end
    return results

def repair_and_extract_json(text):
    """鲁棒的 JSON 解析器，自动处理 Markdown 标记、未闭合尾部、缺失括号及断片提取。"""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    
    # 尝试寻找最后一个完整的 '}' 并补齐 ']'
    last_brace = text.rfind("}")
    if last_brace != -1:
        truncated = text[:last_brace + 1].strip()
        first_bracket = truncated.find("[")
        if first_bracket != -1:
            candidate = truncated[first_bracket:] + "]"
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    
    # 正则独立提取闭合的单个 JSON 对象
    items = []
    for match in re.finditer(r'\{[^{}]*\}', text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            pass
    if items:
        return items
    raise ValueError(f"无法解析云端返回的 JSON 数据 (字符长度: {len(text)})")

def validate_llm_realign(items, source_text, max_chars, media_duration):
    if not isinstance(items, list) or not items:
        raise ValueError("云端返回不是非空的 JSON 数组")

    allowed_tags = {"#金句", "#核心观点", "#需B-Roll", "#废话冗余", "#普通", "#引言", "#争议点"}
    validated = []
    previous_start = -1.0
    reconstructed_text = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("t", item.get("text", ""))).strip()
        if not text:
            continue
        start = float(item.get("s", item.get("start", -1)))
        end = float(item.get("e", item.get("end", -1)))
        tag = str(item.get("tag", "#普通"))
        if tag not in allowed_tags:
            tag = "#普通"
        if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end <= start:
            continue
        if media_duration and end > media_duration + 2.0:
            end = min(end, media_duration)
        if start < previous_start:
            start = previous_start
        if end <= start:
            end = round(start + 0.1, 2)
        previous_start = start
        reconstructed_text.append(text)

        # 若单句偶发超出 max_chars，由兜底算法平滑自动二次切分，绝不崩溃中断
        sub_segs = split_oversized_segment(text, start, end, tag, max_chars)
        validated.extend(sub_segs)

    # 字符一致性比对：剥离标点与空格后计算序列相似度，允许微小格式/虚词容差
    clean_reconstructed = strip_punctuation_and_spaces("".join(reconstructed_text))
    clean_source = strip_punctuation_and_spaces(source_text)

    if clean_reconstructed != clean_source:
        similarity = difflib.SequenceMatcher(None, clean_source, clean_reconstructed).ratio() if clean_source else 1.0
        if similarity < 0.85:
            raise ValueError(
                f"云端审校结果偏离原始台词过大（文本相似度仅为 {similarity * 100:.1f}%，低于 85% 阈值）。"
            )
    return validated

def llm_deep_realign(api_key, df, max_chars=14, media_duration=None, batch_size=25):
    """吸收 SubFix 分批调度与发音守恒锁设计，将长音频字幕分批处理并容灾保底，彻底避免单次 token 溢出截断。"""
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if df.empty:
        return df

    total_rows = len(df)
    total_batches = math.ceil(total_rows / batch_size)
    all_reconstructed_items = []

    progress_bar = None
    try:
        progress_bar = st.progress(0.0, text=f"正在启动云端智能审校 (共 {total_rows} 句, {total_batches} 批)...")
    except Exception:
        pass

    for b_idx in range(total_batches):
        start_row = b_idx * batch_size
        end_row = min(total_rows, (b_idx + 1) * batch_size)
        batch_df = df.iloc[start_row:end_row]

        batch_start_s = float(batch_df.iloc[0]["开始(s)"])
        batch_end_s = float(batch_df.iloc[-1]["结束(s)"])
        batch_source_text = "".join(str(r["台词内容"]) for _, r in batch_df.iterrows())

        if progress_bar:
            progress_bar.progress(
                b_idx / total_batches,
                text=f"AI 智能审校中 (第 {b_idx + 1}/{total_batches} 批: {batch_start_s:.1f}s - {batch_end_s:.1f}s)..."
            )

        raw_payload = []
        for _, r in batch_df.iterrows():
            raw_payload.append({
                "s": float(r["开始(s)"]),
                "e": float(r["结束(s)"]),
                "t": str(r["台词内容"])
            })

        prompt = f"""你是一名资深影视纪录片剪辑师。以下是当前对白切片（时码区间: {batch_start_s:.2f}s - {batch_end_s:.2f}s）。
任务与铁律：
1. 【发音守恒锁】ASR 只会声学听错，绝对不会自己近义词替换！严禁润色或擅自更换用词（例如严禁将'然后'改成'接着'），严禁加字减字。如果原文是口语，保留口语；仅在发音相近的同音别字处修正；
2. 梳理中文自然口语，将破碎断句重新黏合成通顺句子；
3. 对超过 {max_chars} 字的句子在自然语意处合理断句；每句不得超过 {max_chars} 字；
4. 合理计算起止时码（s为开始秒，e为结束秒），保证时码在 [{batch_start_s:.2f}, {batch_end_s:.2f}] 内严格单调递增；
5. 打上标签（#金句, #核心观点, #需B-Roll, #废话冗余, #普通）；
6. 严格只输出纯 JSON 数组，严禁包含任何解释或 Markdown 格式代码块！

数据：
{json.dumps(raw_payload, ensure_ascii=False)}
"""
        use_fallback = False
        batch_items = []
        try:
            resp = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096
            )
            raw_reply = resp.choices[0].message.content or ""
            batch_items = repair_and_extract_json(raw_reply)
        except Exception as err:
            try:
                st.warning(f"第 {b_idx + 1} 批云端响应异常（{str(err)}），已自动切换为本地启发式无损算法保底。")
            except Exception:
                pass
            use_fallback = True

        if not use_fallback:
            reconstructed_txt = "".join(str(it.get("t", it.get("text", ""))) for it in batch_items)
            clean_rec = strip_punctuation_and_spaces(reconstructed_txt)
            clean_src = strip_punctuation_and_spaces(batch_source_text)
            sim = difflib.SequenceMatcher(None, clean_src, clean_rec).ratio() if clean_src else 1.0
            if sim < 0.85:
                try:
                    st.warning(f"第 {b_idx + 1} 批云端文本相似度偏低 ({sim * 100:.1f}%)，为防止文字失真已自动采用本地启发式保底。")
                except Exception:
                    pass
                use_fallback = True

        if use_fallback:
            fallback_df = local_semantic_realign(batch_df, max_chars)
            for _, r in fallback_df.iterrows():
                all_reconstructed_items.append({
                    "start": float(r["开始(s)"]),
                    "end": float(r["结束(s)"]),
                    "tag": str(r.get("标签", "#普通")),
                    "text": str(r["台词内容"]),
                    "keep": bool(r.get("保留", True))
                })
        else:
            try:
                validated_batch = validate_llm_realign(batch_items, batch_source_text, max_chars, batch_end_s)
                all_reconstructed_items.extend(validated_batch)
            except Exception as val_err:
                try:
                    st.warning(f"第 {b_idx + 1} 批云端文本守恒校验未通过 ({val_err})，自动采用本地启发式无损保底。")
                except Exception:
                    pass
                fallback_df = local_semantic_realign(batch_df, max_chars)
                for _, r in fallback_df.iterrows():
                    all_reconstructed_items.append({
                        "start": float(r["开始(s)"]),
                        "end": float(r["结束(s)"]),
                        "tag": str(r.get("标签", "#普通")),
                        "text": str(r["台词内容"]),
                        "keep": bool(r.get("保留", True))
                    })

    if progress_bar:
        progress_bar.progress(1.0, text="AI 语义缝合与时码校准全部完成！")
        time.sleep(0.3)
        progress_bar.empty()

    reconstructed = []
    for it in all_reconstructed_items:
        keep_status = it.get("keep")
        if keep_status is None:
            keep_status = (it["tag"] != "#废话冗余")
        reconstructed.append({
            "保留": bool(keep_status),
            "开始(s)": round(it["start"], 2),
            "结束(s)": round(it["end"], 2),
            "标签": it["tag"],
            "台词内容": it["text"]
        })
    return pd.DataFrame(reconstructed)

# ----------------- 剪辑软件工程导出 -----------------
def generate_fcpxml(df, media_path, fps_val, has_video=True):
    frame_duration = frames_to_fcpx_time(1, fps_val)
    clips_xml = ""
    offset_frames = 0
    rate_num, rate_den, _ = get_fps_spec(fps_val)

    max_sec = 10.0
    if not df.empty:
        try:
            max_sec = max(float(df["结束(s)"].max()), 10.0)
        except Exception:
            max_sec = 3600.0
    total_asset_frames = max(1, int(round((max_sec + 2.0) * rate_num / rate_den)))
    asset_duration = frames_to_fcpx_time(total_asset_frames, fps_val)

    for _, row in df.iterrows():
        if not row["保留"]:
            continue
        start_sec, end_sec = float(row["开始(s)"]), float(row["结束(s)"])
        dur_sec = max(0.1, end_sec - start_sec)
        
        start_f = int(round(start_sec * rate_num / rate_den))
        dur_f = max(1, int(round(dur_sec * rate_num / rate_den)))
        raw_tag = str(row.get("标签", "#普通"))
        raw_text = str(row["台词内容"])
        clip_name = escape(f"[{raw_tag}] {raw_text[:10]}", quote=True)
        marker_value = escape(f"[{raw_tag}] {raw_text}", quote=True)

        if has_video:
            clips_xml += f"""
            <asset-clip ref="r2" offset="{frames_to_fcpx_time(offset_frames, fps_val)}" name="{clip_name}" 
                        start="{frames_to_fcpx_time(start_f, fps_val)}" duration="{frames_to_fcpx_time(dur_f, fps_val)}" format="r1">
                <marker start="{frames_to_fcpx_time(start_f, fps_val)}" duration="{frame_duration}" value="{marker_value}"/>
            </asset-clip>"""
        else:
            clips_xml += f"""
            <asset-clip ref="r2" offset="{frames_to_fcpx_time(offset_frames, fps_val)}" name="{clip_name}" 
                        start="{frames_to_fcpx_time(start_f, fps_val)}" duration="{frames_to_fcpx_time(dur_f, fps_val)}" audioRole="dialogue">
                <marker start="{frames_to_fcpx_time(start_f, fps_val)}" duration="{frame_duration}" value="{marker_value}"/>
            </asset-clip>"""
        offset_frames += dur_f

    if not clips_xml.strip():
        clips_xml = f'<gap offset="0s" name="无保留片段" start="0s" duration="{frame_duration}"/>'

    media_name = escape(os.path.basename(media_path), quote=True)
    media_uri = escape(Path(media_path).resolve().as_uri(), quote=True)
    video_flag = "1" if has_video else "0"
    asset_decl = f'<asset id="r2" name="{media_name}" src="{media_uri}" start="0s" duration="{asset_duration}" format="r1" hasAudio="1" hasVideo="{video_flag}"/>'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
    <resources>
        <format id="r1" name="FFVideoFormat1080p" width="1920" height="1080" frameDuration="{frame_duration}"/>
        {asset_decl}
    </resources>
    <library>
        <event name="AI_RoughCut_Import">
            <project name="RoughCut_Result">
                <sequence format="r1" tcStart="0s" duration="{frames_to_fcpx_time(max(1, offset_frames), fps_val)}">
                    <spine>{clips_xml}</spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

def generate_premiere_xml(df, media_path, fps_val, has_video=True):
    rate_num, rate_den, timebase = get_fps_spec(fps_val)
    ntsc = "TRUE" if rate_den != 1 else "FALSE"
    offset_frames = 0
    clip_id_counter = 1
    media_name = escape(os.path.basename(media_path), quote=True)
    media_uri = escape(Path(media_path).resolve().as_uri(), quote=True)

    max_sec = 10.0
    if not df.empty:
        try:
            max_sec = max(float(df["结束(s)"].max()), 10.0)
        except Exception:
            max_sec = 3600.0
    total_asset_frames = max(1, int(round((max_sec + 2.0) * rate_num / rate_den)))

    if has_video:
        video_clip_items = ""
        audio_clip_items = ""
        for _, row in df.iterrows():
            if not row["保留"]:
                continue
            start_sec, end_sec = float(row["开始(s)"]), float(row["结束(s)"])
            dur_frames = max(1, int(round((end_sec - start_sec) * rate_num / rate_den)))
            in_frame = int(round(start_sec * rate_num / rate_den))
            out_frame = in_frame + dur_frames
            raw_tag = str(row.get("标签", "#普通"))
            raw_text = str(row["台词内容"])
            clip_name = escape(f"[{raw_tag}] {raw_text[:12]}", quote=True)
            
            file_ref = '<file id="file-1"/>'
            if clip_id_counter == 1:
                file_ref = f'''<file id="file-1">
                    <name>{media_name}</name>
                    <pathurl>{media_uri}</pathurl>
                    <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                    <duration>{total_asset_frames}</duration>
                    <media>
                        <video>
                            <samplecharacteristics>
                                <width>1920</width>
                                <height>1080</height>
                                <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                            </samplecharacteristics>
                        </video>
                        <audio>
                            <samplecharacteristics>
                                <depth>16</depth>
                                <samplerate>48000</samplerate>
                            </samplecharacteristics>
                            <channelcount>2</channelcount>
                        </audio>
                    </media>
                </file>'''

            video_clip_items += f"""
            <clipitem id="clipitem-v{clip_id_counter}">
                <name>{clip_name}</name>
                <duration>{dur_frames}</duration>
                <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                <start>{offset_frames}</start>
                <end>{offset_frames + dur_frames}</end>
                <in>{in_frame}</in>
                <out>{out_frame}</out>
                {file_ref}
                <sourcetrack><mediatype>video</mediatype><trackindex>1</trackindex></sourcetrack>
            </clipitem>"""

            audio_clip_items += f"""
            <clipitem id="clipitem-a{clip_id_counter}">
                <name>{clip_name}</name>
                <duration>{dur_frames}</duration>
                <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                <start>{offset_frames}</start>
                <end>{offset_frames + dur_frames}</end>
                <in>{in_frame}</in>
                <out>{out_frame}</out>
                <file id="file-1"/>
                <sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>
            </clipitem>"""
            offset_frames += dur_frames
            clip_id_counter += 1

        media_block = f"""
      <video>
        <format>
          <samplecharacteristics>
            <width>1920</width>
            <height>1080</height>
            <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
          </samplecharacteristics>
        </format>
        <track>{video_clip_items}
        </track>
      </video>
      <audio>
        <numOutputChannels>2</numOutputChannels>
        <format>
          <samplecharacteristics>
            <depth>16</depth>
            <samplerate>48000</samplerate>
          </samplecharacteristics>
        </format>
        <track>{audio_clip_items}
        </track>
      </audio>"""
    else:
        audio_clip_items = ""
        for _, row in df.iterrows():
            if not row["保留"]:
                continue
            start_sec, end_sec = float(row["开始(s)"]), float(row["结束(s)"])
            dur_frames = max(1, int(round((end_sec - start_sec) * rate_num / rate_den)))
            in_frame = int(round(start_sec * rate_num / rate_den))
            out_frame = in_frame + dur_frames
            raw_tag = str(row.get("标签", "#普通"))
            raw_text = str(row["台词内容"])
            clip_name = escape(f"[{raw_tag}] {raw_text[:12]}", quote=True)
            
            file_ref = '<file id="file-1"/>'
            if clip_id_counter == 1:
                file_ref = f'''<file id="file-1">
                    <name>{media_name}</name>
                    <pathurl>{media_uri}</pathurl>
                    <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                    <duration>{total_asset_frames}</duration>
                    <media>
                        <audio>
                            <samplecharacteristics>
                                <depth>16</depth>
                                <samplerate>48000</samplerate>
                            </samplecharacteristics>
                            <channelcount>2</channelcount>
                        </audio>
                    </media>
                </file>'''

            audio_clip_items += f"""
            <clipitem id="clipitem-a{clip_id_counter}">
                <name>{clip_name}</name>
                <duration>{dur_frames}</duration>
                <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
                <start>{offset_frames}</start>
                <end>{offset_frames + dur_frames}</end>
                <in>{in_frame}</in>
                <out>{out_frame}</out>
                {file_ref}
                <sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>
            </clipitem>"""
            offset_frames += dur_frames
            clip_id_counter += 1

        media_block = f"""
      <audio>
        <numOutputChannels>2</numOutputChannels>
        <format>
          <samplecharacteristics>
            <depth>16</depth>
            <samplerate>48000</samplerate>
          </samplecharacteristics>
        </format>
        <track>{audio_clip_items}
        </track>
      </audio>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
  <sequence id="sequence-1">
    <name>AI_RoughCut_Premiere</name>
    <duration>{offset_frames}</duration>
    <rate><timebase>{timebase}</timebase><ntsc>{ntsc}</ntsc></rate>
    <media>{media_block}
    </media>
  </sequence>
</xmeml>"""

def generate_docx(df, media_name, fps_val):
    doc = Document()
    doc.add_heading("🎬 场记台词与粗剪索引表", 0)
    doc.add_paragraph(f"素材文件：{media_name}  |  工程帧率：{fps_val} fps")

    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '时间码区间'
    hdr_cells[1].text = '属性标签'
    hdr_cells[2].text = '台词内容'
    hdr_cells[3].text = '剪辑决策'

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        start_tc = format_timecode(row["开始(s)"], fps_val)
        end_tc = format_timecode(row["结束(s)"], fps_val)
        row_cells[0].text = f"{start_tc}\n➔ {end_tc}"
        row_cells[1].text = str(row.get("标签", "#普通"))
        row_cells[2].text = str(row["台词内容"])
        row_cells[3].text = "保留" if row["保留"] else "剔除"

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def generate_srt(df):
    srt_content = ""
    counter = 1
    for _, row in df.iterrows():
        if not row.get("保留", True):
            continue
        start_tc = format_timecode_ms(float(row["开始(s)"]), ',')
        end_tc = format_timecode_ms(float(row["结束(s)"]), ',')
        text = str(row["台词内容"]).strip()
        if text:
            srt_content += f"{counter}\n{start_tc} --> {end_tc}\n{text}\n\n"
            counter += 1
    return srt_content

def generate_vtt(df):
    vtt_content = "WEBVTT\n\n"
    for _, row in df.iterrows():
        if not row.get("保留", True):
            continue
        start_tc = format_timecode_ms(float(row["开始(s)"]), '.')
        end_tc = format_timecode_ms(float(row["结束(s)"]), '.')
        text = str(row["台词内容"]).strip()
        if text:
            vtt_content += f"{start_tc} --> {end_tc}\n{text}\n\n"
    return vtt_content


# ----------------- 界面主逻辑 -----------------
main_logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="width:52px; height:52px; border-radius:12px; margin-right:16px; box-shadow: 0 4px 16px rgba(59,130,246,0.3); vertical-align: middle;">' if LOGO_B64 else ''
st.markdown(f"""
    <div class="brand-header" style="display: flex; align-items: center;">
        {main_logo_html}
        <div class="brand-text">
            <h1 style="margin: 0;">CutPilot AI <span style="font-size: 1.1rem; font-weight: 600; color: #3b82f6; vertical-align: middle; margin-left: 8px;">v1.6</span></h1>
            <p style="margin: 4px 0 0 0;">本地大模型 & 阿里百炼 / 火山引擎 • 智能场记与粗剪工作流</p>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("<div class='section-title'>1. 载入原始素材</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("上传音视频素材", type=["mp4", "mov", "wav", "mp3", "m4a"], label_visibility="collapsed")

    if uploaded_file:
        file_unique_id = getattr(uploaded_file, "id", getattr(uploaded_file, "file_id", None))
        upload_id = (file_unique_id, uploaded_file.name, uploaded_file.size)
        if st.session_state.get("_upload_id") != upload_id:
            cleanup_session_temp_files()
            for key in ("cut_data", "media_name", "media_path", "media_duration", "audio_b64", "audio_path", "has_video", "audio_preview_path", "audio_url"):
                st.session_state.pop(key, None)
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_media:
                uploaded_file.seek(0)
                shutil.copyfileobj(uploaded_file, tmp_media)
                local_media_path = track_temp_file(tmp_media.name)
            st.session_state["_upload_id"] = upload_id
            st.session_state["media_path"] = local_media_path
            st.session_state["media_name"] = uploaded_file.name
            stream_info = detect_media_streams(local_media_path)
            st.session_state["has_video"] = stream_info.get("has_video", True)
        else:
            local_media_path = st.session_state["media_path"]

        active_engine = app_cfg.get("engine_type", "local")
        if active_engine == "dashscope":
            btn_title = "🚀 启动阿里百炼语音识别 (云端高速)"
            btn_tip = "☁️ 当前选用【阿里百炼云端引擎】，将自动调用百炼平台语音模型处理"
        elif active_engine == "volcengine":
            btn_title = "🚀 启动火山引擎语音识别 (云端高速)"
            btn_tip = "🌋 当前选用【火山引擎豆包语音】，将自动调用字节跳动语音大模型极速处理"
        else:
            btn_title = "🚀 启动本地语音识别引擎 (纯本地)"
            btn_tip = f"💻 当前选用本地模型【{selected_res['title']}】离线张量计算"

        col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
        with col_upload2:
            st.write("")
            if active_engine == "local" and not is_sel_ready:
                st.info(f"💡 提示：当前选中的【{selected_res['title']}】尚未下载。首次启动将自动从最快服务器极速拉取并缓存，或可先在左侧边栏展开面板预先下载。")
            else:
                st.caption(btn_tip)
                
            if st.button(btn_title, type="primary", use_container_width=True):
                try:
                    tmp_wav = st.session_state.get("audio_path")
                    if not tmp_wav or not os.path.exists(tmp_wav):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                            tmp_wav = track_temp_file(tmp_wav_file.name)
                        with st.spinner("正在提取音频特征..."):
                            extract_audio(local_media_path, tmp_wav)
                    total_dur = get_audio_duration(tmp_wav)
                    if total_dur <= 0:
                        raise RuntimeError("无法读取音频时长，请确认素材含有可解码的音轨")

                    if active_engine == "dashscope":
                        ds_key = app_cfg.get("dashscope_api_key", "").strip()
                        if not ds_key:
                            st.error("请先在左侧边栏【语音引擎配置】中填入 DashScope API Key！")
                            st.stop()
                        ds_model = app_cfg.get("dashscope_model", "paraformer-v2")
                        with st.spinner(f"正在通过阿里百炼 ({ds_model}) 进行高精度语音转写..."):
                            raw_segments = transcribe_dashscope(ds_key, tmp_wav, total_dur, model=ds_model)
                    elif active_engine == "volcengine":
                        vc_key = app_cfg.get("volc_api_key", "").strip()
                        if not vc_key:
                            st.error("请先在左侧边栏【语音引擎配置】中填入火山引擎 API Key / Access Token！")
                            st.stop()
                        vc_appid = app_cfg.get("volc_app_id", "").strip()
                        vc_res = app_cfg.get("volc_resource_id", "volc.bigasr.auc_turbo").strip()
                        with st.spinner(f"正在通过火山引擎豆包语音大模型进行识别..."):
                            raw_segments = transcribe_volcengine(vc_key, vc_appid, vc_res, tmp_wav, total_dur)
                    else:
                        if not is_sel_ready:
                            dl_progress_bar = st.progress(0.0, text="0%")
                            capture = TqdmProgressCapture(dl_progress_bar)
                            sys.stderr = capture
                            try:
                                execute_resource_download(selected_res_key, "auto")
                            finally:
                                sys.stderr = capture.original_stderr
                            dl_progress_bar.empty()

                        with st.spinner(f"正在调度 {current_model_id} 进行张量推理..."):
                            model = load_local_model(current_model_id)
                            raw_segments = run_base_transcribe(model, tmp_wav, total_dur)

                    if not raw_segments:
                        raise RuntimeError("未识别到台词；请检查音轨、模型或素材语言")
                    st.session_state["cut_data"] = pd.DataFrame(raw_segments)
                    st.session_state["media_name"] = uploaded_file.name
                    st.session_state["media_path"] = local_media_path
                    st.session_state["media_duration"] = total_dur
                    st.session_state["audio_path"] = tmp_wav
                    
                    # 生成轻量预览音频（48k单声道 MP3），大幅缩减体积并加速网页播放
                    preview_audio = generate_preview_audio(tmp_wav)
                    st.session_state["audio_preview_path"] = preview_audio
                    preview_mime = "audio/mp3" if preview_audio.endswith(".mp3") else "audio/wav"
                    st.session_state["audio_mime"] = preview_mime

                    # 注册 Streamlit 原生 HTTP Range 分段流式端点
                    st.session_state["audio_url"] = get_streamable_media_url(preview_audio, preview_mime)

                    # 若小于 30MB，同时缓存 Base64 作为双重冗余保障
                    if os.path.exists(preview_audio) and os.path.getsize(preview_audio) <= 30 * 1024 * 1024:
                        with open(preview_audio, "rb") as f:
                            st.session_state["audio_b64"] = base64.b64encode(f.read()).decode("utf-8")
                    else:
                        st.session_state["audio_b64"] = ""
                    st.success("转写完毕！已载入时间码与台词。")
                except Exception as e:
                    st.error(f"转写失败: {str(e)}")

# ----------------- 交互与精修模块 -----------------
if "cut_data" in st.session_state:
    
    with st.container(border=True):
        st.markdown("<div class='section-title'>2. 逐字校对与跟读</div>", unsafe_allow_html=True)

        subs_items_html = ""
        for idx, row in st.session_state["cut_data"].iterrows():
            s = float(row["开始(s)"])
            e = float(row["结束(s)"])
            tag = escape(str(row.get("标签", "#普通")))
            txt = escape(str(row["台词内容"]))
            subs_items_html += f"""
            <div class="sub-row" data-start="{s}" data-end="{e}" onclick="seekTo({s})">
                <span class="sub-tc">[{format_timecode(s, fps)} - {format_timecode(e, fps)}]</span>
                <span class="sub-tag">{tag}</span>
                <span class="sub-txt">{txt}</span>
            </div>
            """

        audio_path = st.session_state.get("audio_preview_path") or st.session_state.get("audio_path", "")
        audio_mime = st.session_state.get("audio_mime", "audio/mp3")
        audio_url = st.session_state.get("audio_url", "")
        audio_b64 = st.session_state.get("audio_b64", "")

        # 动态补齐流式与 Base64 音频源：每次 rerun 主动向 Streamlit 媒体管理器续期登记，杜绝 404
        if audio_path and os.path.exists(audio_path):
            audio_url = get_streamable_media_url(audio_path, audio_mime)
            st.session_state["audio_url"] = audio_url
            if not audio_b64 and os.path.getsize(audio_path) <= 30 * 1024 * 1024:
                try:
                    with open(audio_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                        st.session_state["audio_b64"] = audio_b64
                except Exception:
                    pass

        source_tags = ""
        if audio_url:
            source_tags += f'<source src="{audio_url}" type="{audio_mime}">\n'
        if audio_b64:
            source_tags += f'<source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}">\n'

        is_light_theme = ("浅色" in theme_choice)
        is_follow_sys = ("跟随系统" in theme_choice)

        p_bg = "#ffffff" if is_light_theme else "#0f172a"
        p_border = "#e2e8f0" if is_light_theme else "#334155"
        p_sub_bg = "#f8fafc" if is_light_theme else "#1e293b"
        p_audio_filter = "none" if is_light_theme else "invert(0.9) hue-rotate(180deg)"
        p_sub_text = "#334155" if is_light_theme else "#94a3b8"
        p_sub_hover_text = "#0f172a" if is_light_theme else "#f8fafc"
        p_sub_hover_bg = "#f1f5f9" if is_light_theme else "rgba(255, 255, 255, 0.05)"
        p_sub_border_hover = "#cbd5e1" if is_light_theme else "rgba(255, 255, 255, 0.1)"
        p_active_bg = "#eff6ff" if is_light_theme else "rgba(37, 99, 235, 0.22)"
        p_active_border = "#2563eb" if is_light_theme else "#3b82f6"
        p_active_text = "#1d4ed8" if is_light_theme else "#ffffff"
        p_tag_bg = "#e2e8f0" if is_light_theme else "#334155"
        p_tag_text = "#475569" if is_light_theme else "#cbd5e1"
        p_scrollbar_thumb = "#cbd5e1" if is_light_theme else "#475569"

        sys_player_media = """
        @media (prefers-color-scheme: light) {
            #playerWrapper { background: #ffffff !important; border-color: #e2e8f0 !important; }
            #continuousPlayer { filter: none !important; }
            #subContainer { background: #f8fafc !important; border-color: #e2e8f0 !important; }
            #subContainer::-webkit-scrollbar-thumb { background: #cbd5e1 !important; }
            .sub-row { color: #334155 !important; background: transparent !important; }
            .sub-row:hover { background: #f1f5f9 !important; color: #0f172a !important; border-color: #cbd5e1 !important; }
            .sub-row.active { background: #eff6ff !important; border-left: 4px solid #2563eb !important; color: #1d4ed8 !important; }
            .sub-tag { background: #e2e8f0 !important; color: #475569 !important; }
        }
        """ if is_follow_sys else ""

        player_html = f"""
        <div id="playerWrapper" style="background:{p_bg}; border-radius:12px; padding:18px; border: 1px solid {p_border}; font-family:system-ui, -apple-system, sans-serif;">
            <audio id="continuousPlayer" controls style="width:100%; height:45px; outline:none; margin-bottom:12px; filter:{p_audio_filter};">
                {source_tags}
            </audio>
            
            <div id="subContainer" style="max-height:280px; overflow-y:auto; padding:12px; border-radius:8px; background:{p_sub_bg}; border:1px solid {p_border};">
                {subs_items_html}
            </div>
        </div>

        <style>
            /* 滚动条美化 */
            #subContainer::-webkit-scrollbar {{ width: 8px; }}
            #subContainer::-webkit-scrollbar-track {{ background: {p_sub_bg}; border-radius: 4px; }}
            #subContainer::-webkit-scrollbar-thumb {{ background: {p_scrollbar_thumb}; border-radius: 4px; }}

            .sub-row {{
                padding: 10px 14px;
                margin-bottom: 8px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                color: {p_sub_text};
                font-size: 15px;
                display: flex;
                align-items: flex-start;
                gap: 16px;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid transparent;
            }}
            .sub-row:hover {{
                background: {p_sub_hover_bg};
                border-color: {p_sub_border_hover};
                color: {p_sub_hover_text};
            }}
            .sub-row.active {{
                background: {p_active_bg} !important;
                border-left: 4px solid {p_active_border} !important;
                color: {p_active_text} !important;
                font-weight: 600 !important;
                transform: translateX(4px);
            }}
            .sub-tc {{
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 13px;
                color: #64748b;
                flex-shrink: 0;
                margin-top: 2px;
            }}
            .sub-tag {{
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 12px;
                background: {p_tag_bg};
                color: {p_tag_text};
                flex-shrink: 0;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            .sub-txt {{
                flex-grow: 1;
                line-height: 1.5;
            }}
            {sys_player_media}
        </style>

        <script>
            const audio = document.getElementById('continuousPlayer');
            const rows = document.querySelectorAll('.sub-row');
            const container = document.getElementById('subContainer');

            audio.ontimeupdate = function() {{
                const cur = audio.currentTime;
                rows.forEach(r => {{
                    const st = parseFloat(r.getAttribute('data-start'));
                    const et = parseFloat(r.getAttribute('data-end'));
                    if (cur >= st && cur <= et) {{
                        if (!r.classList.contains('active')) {{
                            rows.forEach(x => x.classList.remove('active'));
                            r.classList.add('active');
                            r.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
                    }}
                }});
            }};

            function seekTo(sec) {{
                audio.currentTime = sec;
                audio.play();
            }}
        </script>
        """
        components.html(player_html, height=400)
        with st.expander("📋 查看字幕数据明细表 (含时码与剪辑标签)", expanded=False):
            st.dataframe(
                st.session_state["cut_data"][["开始(s)", "结束(s)", "标签", "台词内容"]],
                use_container_width=True,
                hide_index=True,
                height=260,
            )

    # ----------------- AI 语义缝合与时间码精修 -----------------
    with st.container(border=True):
        st.markdown("<div class='section-title'>3. AI 语义缝合与时间码精修</div>", unsafe_allow_html=True)
        
        col_ai1, col_ai2 = st.columns([3, 2])
        with col_ai1:
            realign_mode = st.radio(
                "选择 AI 语义黏合引擎：", 
                ["⚡ 本地启发式语法对齐 (离线极速)", "🧠 云端 LLM 深度审校 (需百炼 API Key)"],
                index=0,
                horizontal=True
            )
            if "云端" in realign_mode:
                default_opt_key = app_cfg.get("dashscope_api_key", "")
                opt_key = st.text_input("DashScope API Key", value=default_opt_key, type="password", placeholder="sk-...", help="自动同步侧边栏已保存的百炼 Key")
        with col_ai2:
            st.markdown("<div style='padding: 8px 12px; background: rgba(59, 130, 246, 0.08); border-radius: 8px; color: var(--primary-accent); font-size: 0.85rem; margin-bottom: 8px;'>💡 算法基于语意停顿与语法结构自然缝合长短句，原汁原味保留全部字词（0删词），并严密重算时间戳。</div>", unsafe_allow_html=True)
            if st.button("✨ 启动 AI 语义缝合与时间码精修", type="primary", use_container_width=True):
                try:
                    if "本地" in realign_mode:
                        with st.spinner("正在本地快速重组上下文语意与时码对齐..."):
                            st.session_state["cut_data"] = local_semantic_realign(st.session_state["cut_data"], target_max_chars)
                    else:
                        if not opt_key:
                            st.error("请填入 DashScope API Key！")
                            st.stop()
                        st.session_state["cut_data"] = llm_deep_realign(
                            opt_key,
                            st.session_state["cut_data"],
                            target_max_chars,
                            st.session_state.get("media_duration"),
                        )
                    st.success("AI 语义缝合与时码校准完成！")
                    st.rerun()
                except Exception as e:
                    st.error(f"校准失败: {str(e)}")

    # ----------------- 错别字与数据微调 -----------------
    col_L, col_R = st.columns(2)
    with col_L:
        with st.container(border=True):
            st.markdown("<div class='section-title'>🔍 批量词汇清洗</div>", unsafe_allow_html=True)
            search_kw = st.text_input("查找错字 / 废话口癖", placeholder="例如：就是说")
            target_kw = st.text_input("替换为 (留空则默认删除)", placeholder="留空即为【删除】该词")
            
            if search_kw.strip():
                if st.button("执行清洗", type="secondary"):
                    is_delete_mode = (target_kw.strip() == "")
                    apply_count = 0
                    for idx, row in st.session_state["cut_data"].iterrows():
                        cur_t = str(row["台词内容"])
                        if search_kw in cur_t:
                            st.session_state["cut_data"].at[idx, "台词内容"] = cur_t.replace(search_kw, target_kw.strip()) if not is_delete_mode else cur_t.replace(search_kw, "")
                            apply_count += 1
                    st.success(f"已处理 {apply_count} 处匹配！")
                    st.rerun()
            
    with col_R:
        with st.container(border=True):
            st.markdown("<div class='section-title'>🧹 格式净化</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.9rem; opacity:0.8;'>专业视频后期往往不需要标点符号。使用此功能一键剥离所有中英标点。</p>", unsafe_allow_html=True)
            if st.button("一键剔除全篇标点符号", type="secondary", use_container_width=True):
                def remove_punct(txt):
                    return re.sub(r"[，。！？、“”‘’：；,.!?:;\n]", "", str(txt)).strip()
                st.session_state["cut_data"]["台词内容"] = st.session_state["cut_data"]["台词内容"].apply(remove_punct)
                st.success("标点清理完毕！")
                st.rerun()

    # ----------------- 最终场记编辑表格 -----------------
    with st.container(border=True):
        st.markdown("<div class='section-title'>📋 总控制台对齐表</div>", unsafe_allow_html=True)
        st.session_state["cut_data"] = st.data_editor(
            st.session_state["cut_data"],
            column_config={
                "保留": st.column_config.CheckboxColumn("保留", default=True, width="small"),
                "标签": st.column_config.SelectboxColumn("叙事标签", options=["#金句", "#核心观点", "#废话冗余", "#需B-Roll", "#引言", "#争议点"], width="medium"),
                "开始(s)": st.column_config.NumberColumn("入点(s)", format="%.2f", width="small"),
                "结束(s)": st.column_config.NumberColumn("出点(s)", format="%.2f", width="small"),
                "台词内容": st.column_config.TextColumn("台词内容", width="large"),
            },
            use_container_width=True,
            hide_index=True,
            height=350
        )

    # ----------------- 导出模块 -----------------
    with st.container(border=True):
        st.markdown("<div class='section-title'>📦 工程与交付物分发</div>", unsafe_allow_html=True)
        
        tab_nle, tab_doc, tab_sub = st.tabs(["🎞️ 剪辑软件工程", "📄 文档场记", "📜 字幕序列"])

        with tab_nle:
            st.markdown("<br/>", unsafe_allow_html=True)
            col_n1, col_n2, col_n3 = st.columns(3)
            has_video = st.session_state.get("has_video", True)
            with col_n1:
                if st.button("🍏 导出 Final Cut Pro XML", use_container_width=True):
                    fcpxml_str = generate_fcpxml(st.session_state["cut_data"], st.session_state["media_path"], fps, has_video=has_video)
                    save_file_locally(fcpxml_str, f"{os.path.splitext(st.session_state['media_name'])[0]}_FCP", ".fcpxml")
            with col_n2:
                if st.button("🎬 导出 Premiere Pro XML", use_container_width=True):
                    pr_xml_str = generate_premiere_xml(st.session_state["cut_data"], st.session_state["media_path"], fps, has_video=has_video)
                    save_file_locally(pr_xml_str, f"{os.path.splitext(st.session_state['media_name'])[0]}_Premiere", ".xml")
            with col_n3:
                if st.button("✨ 导出 DaVinci Resolve XML", use_container_width=True):
                    # DaVinci Resolve can import FCPXML smoothly
                    resolve_xml_str = generate_fcpxml(st.session_state["cut_data"], st.session_state["media_path"], fps, has_video=has_video)
                    save_file_locally(resolve_xml_str, f"{os.path.splitext(st.session_state['media_name'])[0]}_Resolve", ".fcpxml")

        with tab_doc:
            st.markdown("<br/>", unsafe_allow_html=True)
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                if st.button("📝 导出 纯台词文本 (.txt)", use_container_width=True):
                    pure_txt = "\n".join(st.session_state["cut_data"][st.session_state["cut_data"]["保留"]]["台词内容"].tolist())
                    save_file_locally(pure_txt, f"{os.path.splitext(st.session_state['media_name'])[0]}_dialogue", ".txt")
            with col_d2:
                if st.button("⏱️ 导出 时码场记本 (.txt)", use_container_width=True):
                    tc_txt = ""
                    for _, r in st.session_state["cut_data"].iterrows():
                        if r["保留"]:
                            tc_txt += f"[{format_timecode(r['开始(s)'], fps)} - {format_timecode(r['结束(s)'], fps)}] [{r.get('标签', '')}] {r['台词内容']}\n"
                    save_file_locally(tc_txt, f"{os.path.splitext(st.session_state['media_name'])[0]}_timecode", ".txt")
            with col_d3:
                if st.button("📘 导出 排版场记表 (.docx)", use_container_width=True):
                    docx_bytes = generate_docx(st.session_state["cut_data"], st.session_state["media_name"], fps)
                    save_file_locally(docx_bytes, f"{os.path.splitext(st.session_state['media_name'])[0]}_场记表", ".docx", is_bytes=True)
                
        with tab_sub:
            st.markdown("<br/>", unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("🎬 导出 标准字幕 SRT", use_container_width=True):
                    srt_str = generate_srt(st.session_state["cut_data"])
                    save_file_locally(srt_str, f"{os.path.splitext(st.session_state['media_name'])[0]}", ".srt")
            with col_s2:
                if st.button("🌐 导出 Web字幕 VTT", use_container_width=True):
                    vtt_str = generate_vtt(st.session_state["cut_data"])
                    save_file_locally(vtt_str, f"{os.path.splitext(st.session_state['media_name'])[0]}", ".vtt")
