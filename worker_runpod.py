import torch
import numpy as np
from PIL import Image
import totoro
import nodes
import sys
sys.path.append('/content/TotoroUI/IPAdapter')
import IPAdapterPlus
sys.path.append('/content/TotoroUI/InstantID')
import InstantID
import scipy
import model_management
from latent_resizer import LatentResizer
from totoro import model_management
import gc
import random
import os, json, requests, runpod

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

def upscale(latent, upscale):
  device = model_management.get_torch_device()
  samples = latent.to(device=device, dtype=torch.float16)
  model = LatentResizer.load_model('/content/TotoroUI/models/sdxl_resizer.pt', device, torch.float16)
  model.to(device=device)
  latent_out = (model(0.13025 * samples, scale=upscale) / 0.13025)
  latent_out = latent_out.to(device="cpu")
  model.to(device=model_management.vae_offload_device())
  return ({"samples": latent_out},)

def download_file(url, save_dir='/content/TotoroUI/models'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

with torch.no_grad():
    model_patcher, clip, vae, clipvision = totoro.sd.load_checkpoint_guess_config("/content/TotoroUI/models/raemuXL_v35Lightning.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
    IPAdapterPlus_model = IPAdapterPlus.IPAdapterUnifiedLoader().load_models(model_patcher, 'PLUS FACE (portraits)', lora_strength=0.0, provider="CUDA")
    instantid = InstantID.InstantIDModelLoader().load_model("/content/TotoroUI/models/instantid/ip-adapter.bin")
    insightface = InstantID.InstantIDFaceAnalysis().load_insight_face("CUDA")
    instantid_control_net = totoro.controlnet.load_controlnet("/content/TotoroUI/models/controlnet/SDXL/instantid/diffusion_pytorch_model.safetensors")

@torch.inference_mode()
def generate(input):
    values = input["input"]
    input_image = values['input_image']
    kps_image = values['kps_image']
    dw_image = values['dw_image']
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    input_image = download_file(input_image)
    kps_image = download_file(kps_image)
    dw_image = download_file(dw_image)
    output_image, output_mask = nodes.LoadImage().load_image(input_image) 
    image_kps, image_kps_mask = nodes.LoadImage().load_image(kps_image)
    image_dw, image_dw_mask = nodes.LoadImage().load_image(dw_image)
    ip_model_patcher = IPAdapterPlus.IPAdapterAdvanced().apply_ipadapter(IPAdapterPlus_model[0], IPAdapterPlus_model[1], image=output_image, weight_type="style transfer")
    tokens = clip.tokenize(positive_prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    n_tokens = clip.tokenize(negative_prompt)
    n_cond, n_pooled = clip.encode_from_tokens(n_tokens, return_pooled=True)
    n_cond = [[n_cond, {"pooled_output": n_pooled}]]
    work_model, instantid_cond, instantid_n_cond = InstantID.ApplyInstantID().apply_instantid(instantid=instantid[0], insightface=insightface[0], control_net=instantid_control_net, image=output_image, model=ip_model_patcher[0], positive=cond, negative=n_cond, start_at=0.0, end_at=1.0, weight=0.80, image_kps=image_kps)
    openpose_control_net = totoro.controlnet.load_controlnet("/content/TotoroUI/models/controlnet/thibaud_xl_openpose.safetensors")
    openpose_cond = nodes.ControlNetApply().apply_controlnet(conditioning=instantid_cond, control_net=openpose_control_net, image=image_dw, strength=0.90)

    latent = {"samples":torch.zeros([1, 4, 1024 // 8, 1024 // 8])}
    ran = random.randint(0, 65535)
    print(ran)
    sample = nodes.common_ksampler(model=work_model, 
                          seed=ran, 
                          steps=4, 
                          cfg=1.3, 
                          sampler_name="dpmpp_sde_gpu", 
                          scheduler="karras", 
                          positive=openpose_cond[0], 
                          negative=instantid_n_cond,
                          latent=latent, 
                          denoise=0.95)

    with torch.inference_mode():
        sample = sample[0]["samples"].to(torch.float16)
        vae.first_stage_model.cuda()
        decoded = vae.decode_tiled(sample).detach()

    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/output_image.png")

    result = "/content/output_image.png"

    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})