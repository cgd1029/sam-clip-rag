import os
import tempfile

import streamlit as st

from core.singing_synth import extract_timbre_profile, synthesize_singing

st.set_page_config(page_title="音色学习 AI 唱歌", page_icon="🎤", layout="centered")

st.title("🎤 输入录音 -> 学习音色 -> AI 唱歌")
st.markdown(
    "上传一段参考录音，然后输入乐谱（音符:时长秒），即可合成一段该音色的歌声（MVP 原型）。"
)

ref_audio = st.file_uploader("1) 上传参考录音 (wav/mp3)", type=["wav", "mp3", "ogg", "flac"])
score = st.text_area(
    "2) 输入乐谱（格式: C4:0.5, D4:0.5, E4:1.0）",
    value="C4:0.5, D4:0.5, E4:1.0, G4:1.0, E4:0.5, D4:0.5, C4:1.0",
    height=80,
)
lyrics = st.text_input("歌词（可选，当前仅占位）", value="la la la")
tempo = st.slider("速度倍率", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

if st.button("开始 AI 唱歌"):
    if ref_audio is None:
        st.error("请先上传参考录音")
    elif not score.strip():
        st.error("请填写乐谱")
    else:
        with st.spinner("正在分析音色并合成..."):
            workdir = tempfile.mkdtemp(prefix="voice_clone_demo_")
            in_path = os.path.join(workdir, ref_audio.name)
            out_path = os.path.join(workdir, "ai_singing.wav")

            with open(in_path, "wb") as f:
                f.write(ref_audio.read())

            profile = extract_timbre_profile(in_path)
            synthesize_singing(
                profile=profile,
                score=score,
                out_path=out_path,
                lyrics=lyrics,
                tempo_scale=tempo,
            )

            st.success("合成完成")
            st.audio(out_path)
            with open(out_path, "rb") as f:
                st.download_button("下载音频", data=f, file_name="ai_singing.wav", mime="audio/wav")

st.caption("提示：想要更拟真，可在此流程上接入 RVC / So-VITS-SVC。")
