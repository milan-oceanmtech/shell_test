import torch
import uuid
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler

model_path = "./final_modal/1000/"             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

g_cuda = None

g_cuda = torch.Generator(device='cuda')
seed = 52362
g_cuda.manual_seed(seed)

prompt = "Portrait of milantest , hyperdetailed portrait of a stunningly beautiful european boy with short hair androgynous guard made of iridescent metals shiny gems, bright nimbus, thin golden necklace, inspired by ross tran and wlop and masamune shirow and kuvshinov, concept art, intricate, photorealistic, octane render, rtx, hdr, unreal engine, dnd digital art by artgerm"
negative_prompt = ""
num_samples = 4
guidance_scale = 7.5
num_inference_steps = 50
height = 512
width = 512

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    img.save("./final_images/"+str(uuid.uuid4())+".png")
