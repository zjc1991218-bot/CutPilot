import streamlit as st
import streamlit.components.v1 as components
import os

comp_dir = "my_comp"
os.makedirs(comp_dir, exist_ok=True)
with open(f"{comp_dir}/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
      <body>
        <div id="app" style="color: white; background: #333; padding: 20px;">
            Hello from component!
            <button onclick="sendData()">Send Data</button>
        </div>
        <script>
          function sendMessageToStreamlitClient(type, data) {
            var outData = Object.assign({
                isStreamlitMessage: true,
                type: type,
            }, data);
            window.parent.postMessage(outData, "*");
          }
          function sendData() {
            sendMessageToStreamlitClient("setComponentValue", {value: "Clicked!", dataType: "json"});
          }
          window.onload = function() {
            sendMessageToStreamlitClient("setFrameHeight", {height: 100});
            // We must tell streamlit we are ready
            sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1});
          };
          window.addEventListener("message", function(e) {
            if (e.data.type === "streamlit:render") {
                // handle args
                console.log(e.data.args);
            }
          });
        </script>
      </body>
    </html>
    """)

my_comp = components.declare_component("my_comp", path=comp_dir)
res = my_comp(key="foo")
st.write("Result:", res)
