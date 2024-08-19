import os, json, requests, runpod

import sys
sys.path.append('/content/TotoroUI/IPAdapter')
import IPAdapterPlus
sys.path.append('/content/TotoroUI/InstantID')
import InstantID
sys.path.append('/content/TotoroUI/Impact')
import torch
import numpy as np
from PIL import Image
import totoro
import nodes
import detailer
import scipy
import model_management
import gc
import random

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
    model_patcher, clip, vae, clipvision = totoro.sd.load_checkpoint_guess_config("/content/TotoroUI/models/dreamshaperXL_lightningDPMSDE.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
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
    face_positive_prompt = values['face_positive_prompt']
    face_negative_prompt = values['face_negative_prompt']
    width = values['width']
    height = values['height']
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

    latent = {"samples":torch.zeros([1, 4, height // 8, width // 8])}
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

    sample = sample[0]["samples"].to(torch.float16)
    vae.first_stage_model.cuda()
    decoded = vae.decode(sample).detach()

    model = detailer.load_yolo("/content/TotoroUI/models/ultralytics/bbox/Eyes.pt")
    bbox_detector = detailer.UltraBBoxDetector(model)

    bbox_detector.setAux('face')
    segs = bbox_detector.detect(image=decoded, threshold=0.50, dilation=10, crop_factor=3.0, drop_size=10, detailer_hook=None)
    bbox_detector.setAux(None)

    face_tokens = clip.tokenize(face_positive_prompt)
    face_cond, face_pooled = clip.encode_from_tokens(face_tokens, return_pooled=True)
    face_cond = [[face_cond, {"pooled_output": face_pooled}]]
    face_n_tokens = clip.tokenize(face_negative_prompt)
    face_n_cond, face_n_pooled = clip.encode_from_tokens(face_n_tokens, return_pooled=True)
    face_n_cond = [[face_n_cond, {"pooled_output": face_n_pooled}]]

    enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
        detailer.DetailerForEach.do_detail(image=decoded, segs=segs, model=model_patcher, clip=clip, vae=vae, guide_size=384, guide_size_for_bbox=True, max_size=1024, seed=ran, steps=4, cfg=1.0,
                                    sampler_name="dpmpp_sde", scheduler="karras", positive=face_cond, negative=face_n_cond, denoise=0.35, feather=5, noise_mask=True,
                                    force_inpaint=True, wildcard_opt=None, detailer_hook=None, refiner_ratio=0.2, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                                    cycle=1, inpaint_model=False, noise_mask_feather=20)
    upscaled_img = nodes.ImageScaleBy.upscale(_, image=enhanced_img, upscale_method="lanczos", scale_by=1.50)
    adjust_img = detailer.image_filters(image=upscaled_img[0], brightness=0.0, contrast=1.0, saturation=1.0, sharpness=3, blur=0, gaussian_blur=0.0, edge_enhance=0.10, detail_enhance=True)

    Image.fromarray(np.array(adjust_img[0]*255, dtype=np.uint8)[0]).save("/content/output_image.png")

    result = "/content/output_image.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})