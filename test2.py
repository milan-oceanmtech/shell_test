import torch
import uuid
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler

model_path = "./final_modal/800/"             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")


g_cuda = None

g_cuda = torch.Generator(device='cuda')
seed = 445896576
g_cuda.manual_seed(seed)

prompt = "Portrait of milantest, freckles, curly middle part haircut, curly hair, middle part hairstyle, smiling kindly, wearing a bowtie and sweater vest, intricate, elegant, glowing lights, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by wlop, mars ravelo and greg rutkowski"
negative_prompt = ""
num_samples = 5
guidance_scale = 13
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
