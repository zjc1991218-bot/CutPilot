import streamlit as st
import streamlit.components.v1 as components
import os

comp_dir = "custom_player"
os.makedirs(comp_dir, exist_ok=True)
with open(f"{comp_dir}/index.html", "w", encoding="utf-8") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; color: white; background: #0f172a; margin: 0; padding: 10px; }
            input, textarea { background: #1e293b; color: white; border: 1px solid #334155; border-radius: 4px; padding: 4px; }
            .row { display: flex; gap: 10px; margin-bottom: 5px; padding: 5px; border: 1px solid transparent; }
            .row.active { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
            #subs { height: 400px; overflow-y: auto; }
            audio { width: 100%; margin-bottom: 10px; }
        </style>
      </head>
      <body>
        <audio id="audio" controls></audio>
        <button onclick="syncData()">Sync to Python</button>
        <div id="subs"></div>
        
        <script>
            let subsData = [];
            let currentVersion = -1;
            let audioNode = document.getElementById("audio");
            let subsNode = document.getElementById("subs");

            function sendMessage(type, data) {
                window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
            }

            function syncData() {
                sendMessage("setComponentValue", { value: subsData, dataType: "json" });
            }

            function buildUI() {
                subsNode.innerHTML = "";
                subsData.forEach((row, i) => {
                    let div = document.createElement("div");
                    div.className = "row";
                    div.id = "row-" + i;
                    
                    let keep = document.createElement("input");
                    keep.type = "checkbox";
                    keep.checked = row.keep;
                    keep.onchange = e => { row.keep = e.target.checked; syncData(); };

                    let time = document.createElement("span");
                    time.innerText = row.s.toFixed(2) + " - " + row.e.toFixed(2);
                    time.style.cursor = "pointer";
                    time.onclick = () => { audioNode.currentTime = row.s; audioNode.play(); };

                    let tag = document.createElement("input");
                    tag.type = "text";
                    tag.value = row.tag || "";
                    tag.style.width = "60px";
                    tag.onchange = e => { row.tag = e.target.value; syncData(); };

                    let text = document.createElement("input");
                    text.type = "text";
                    text.value = row.text || "";
                    text.style.flexGrow = "1";
                    text.onchange = e => { row.text = e.target.value; syncData(); };

                    div.appendChild(keep);
                    div.appendChild(time);
                    div.appendChild(tag);
                    div.appendChild(text);
                    subsNode.appendChild(div);
                });
            }

            audioNode.ontimeupdate = () => {
                let cur = audioNode.currentTime;
                subsData.forEach((row, i) => {
                    let node = document.getElementById("row-" + i);
                    if (!node) return;
                    if (cur >= row.s && cur <= row.e) {
                        if (!node.classList.contains("active")) {
                            document.querySelectorAll(".row").forEach(n => n.classList.remove("active"));
                            node.classList.add("active");
                            node.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    }
                });
            };

            window.addEventListener("message", e => {
                if (e.data.type === "streamlit:render") {
                    let args = e.data.args;
                    if (args.version !== currentVersion) {
                        currentVersion = args.version;
                        subsData = JSON.parse(JSON.stringify(args.data));
                        audioNode.src = args.audio_src;
                        buildUI();
                    }
                    sendMessage("setFrameHeight", { height: 500 });
                }
            });

            window.onload = () => {
                sendMessage("streamlit:componentReady", { apiVersion: 1 });
            };
        </script>
      </body>
    </html>
    """)

player_comp = components.declare_component("player_comp", path=comp_dir)
import pandas as pd
df = pd.DataFrame([{"s": 0.0, "e": 2.0, "tag": "", "text": "Hello", "keep": True}])

res = player_comp(version=1, data=df.to_dict("records"), audio_src="")
st.write("Result from comp:", res)
