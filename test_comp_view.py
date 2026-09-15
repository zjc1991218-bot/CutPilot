import streamlit as st
import streamlit.components.v1 as components

player_comp = components.declare_component("player_comp", path="custom_player")

st.title("Component Test")

data = [
    {"保留": True, "开始(s)": 0.0, "结束(s)": 2.0, "标签": "#金句", "台词内容": "This is line 1."},
    {"保留": True, "开始(s)": 2.0, "结束(s)": 4.0, "标签": "", "台词内容": "This is line 2."}
]

st.write("Calling component...")
res = player_comp(version=1, data=data, audio_src="")
st.write("Result:", res)
