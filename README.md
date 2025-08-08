## Environment Preparation
```bash
conda create -n ov python=3.11
conda activate ov

# Install OpenVINO
pip install --upgrade --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

pip install optimum-intel[nncf]

pip install fastapi uvicorn openai

# If use open-webui as chat frontend
pip install open-webui

conda activate pygame
set http_proxy=
set https_proxy=
open-webui serve

# If use gradio as chat frontend
pip install gradio


## Models

optimum-cli export openvino --model C:/DeepSeek-R1-Distill-Qwen-1.5B --weight-format int4 --task text-generation-with-past DeepSeek-R1-Distill-Qwen-1.5B-ov

## Run
`lightweight_serving.py` launches the service for the speculative decoding application and provides OpenAI-compatible endpoints (i.e. `/v1/chat/completions` and `/v1/models`).


# Launch lightweight serving backend, repo-id-or-model-path is the parent directory of the two models
# Default address is http://ip:8000/v1

cd C:\Users\Lengda\Documents\ov\openwebui-ov-0516

python lightweight_serving.py --repo-id-or-model-path C:/models/Hunyuan-7B-ov --device GPU


python lightweight_serving.py --repo-id-or-model-path C:/models/Hunyuan-7B-Instruct-npu-ov --device NPU


# If use openwebui, add the OpenAI API http://127.0.0.1:8000/v1 in [Admin Panel->Settings->Connection].

# If use gradio webui
python gradio_webui.py --model-url http://127.0.0.1:8000/v1
```
