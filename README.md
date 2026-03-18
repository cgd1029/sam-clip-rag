<<<<<<< codex/create-ai-singing-app-from-audio-input
# sam-clip-rag

这是一个可运行的 **输入录音文件 -> 学习音色 -> AI 唱歌合成** 的 MVP 应用。

## 功能
- 上传参考人声（干声）
- 自动提取音色特征（LPC 声道包络）
- 输入简谱后自动生成歌声音频
- Web 界面交互（Streamlit）

> 说明：当前版本为轻量原型，重在流程闭环和可运行性。若要更接近真实人声唱法，可在此基础上接入 RVC / So-VITS-SVC。

## 快速开始
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

打开浏览器访问：`http://127.0.0.1:7860`

## 使用方法
1. 上传你的参考录音（建议 5~15 秒、无伴奏、噪声低）。
2. 输入乐谱，格式如下：
   - `C4:0.5, D4:0.5, E4:1.0, G4:1.0`
   - 含义：音符 + 持续时间（秒）
3. 点击「开始 AI 唱歌」，下载合成结果。

## 项目结构
- `app.py`：Streamlit Web UI 与主流程。
- `core/singing_synth.py`：音色提取、乐谱解析、歌声合成核心逻辑。
- `tests/test_singing_synth.py`：基础单元测试。

## 后续可扩展方向
- 歌词到发音（G2P）与音素对齐
- 基于神经声码器的高保真合成
- 接入 RVC / So-VITS-SVC 进行更真实的 singing voice conversion
=======
# 1
>>>>>>> main
