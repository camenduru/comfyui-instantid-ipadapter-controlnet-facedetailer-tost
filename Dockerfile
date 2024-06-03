FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 accelerate==0.30.1 insightface==0.7.3 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0

RUN git clone -b totoro https://github.com/camenduru/ComfyUI /content/TotoroUI
RUN git clone -b totoro_v2 https://github.com/camenduru/ComfyUI_IPAdapter_plus /content/TotoroUI/IPAdapter
RUN git clone -b totoro https://github.com/camenduru/ComfyUI_InstantID /content/TotoroUI/InstantID

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/470847 -d /content/TotoroUI/models -o raemuXL_v35Lightning.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -d /content/TotoroUI/models/clip_vision -o CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors  -d /content/TotoroUI/models/ipadapter -o ip-adapter-plus-face_sdxl_vit-h.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose.safetensors -d /content/TotoroUI/models/controlnet -o thibaud_xl_openpose.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/Ttl/ComfyUi_NNLatentUpscale/raw/master/sdxl_resizer.pt -d /content/TotoroUI/models -o sdxl_resizer.pt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/1k3d68.onnx -d /content/TotoroUI/models/insightface/models/antelopev2 -o 1k3d68.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/2d106det.onnx -d /content/TotoroUI/models/insightface/models/antelopev2 -o 2d106det.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx -d /content/TotoroUI/models/insightface/models/antelopev2 -o genderage.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx -d /content/TotoroUI/models/insightface/models/antelopev2 -o glintr100.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx -d /content/TotoroUI/models/insightface/models/antelopev2 -o scrfd_10g_bnkps.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin -d /content/TotoroUI/models/instantid -o ip-adapter.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors -d /content/TotoroUI/models/controlnet/SDXL/instantid -o diffusion_pytorch_model.safetensors

COPY ./worker_runpod.py /content/TotoroUI/worker_runpod.py
WORKDIR /content/TotoroUI
CMD python worker_runpod.py
