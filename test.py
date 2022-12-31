import subprocess
import requests
import json
import os ,shlex
import uuid

os.mkdir("/workspace/class/")
os.mkdir("/workspace/images/")
os.mkdir("/workspace/final_images/")

data = requests.get("https://ffe0-2401-4900-1f3e-414-557e-c80e-3105-4d4c.in.ngrok.io/get_task").json()

for x in data["images"]:
    f = open('/workspace/images/'+x.split("/")[-1],'wb')
    f.write(requests.get(x).content)
    f.close()


with open("concepts_list.json", "w") as f:
    json.dump(data["concepts_list"], f, indent=4)


liness = 'wget -q https://raw.githubusercontent.com/milan-oceanmtech/shell_test/main/test2.py && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py && pip install -q git+https://github.com/ShivamShrirao/diffusers && pip install -q -U --pre triton && pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort &&accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir="./final_modal" \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=100 \
  --save_interval=10000 \
  --save_sample_prompt="photo of milantest" \
  --concepts_list="concepts_list.json" && python convert_diffusers_to_original_stable_diffusion.py --model_path ./final_modal/100/  --checkpoint_path ./final_modal/100/model.ckpt --half && python test2.py'
for x in liness.split("&&"):
    comads = shlex.split(x)


    process = subprocess.Popen(comads, 
                               stdout=subprocess.PIPE,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline()
        print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            break





import requests

for x in os.listdir("./final_images"):

    url = "https://ffe0-2401-4900-1f3e-414-557e-c80e-3105-4d4c.in.ngrok.io/upload"

    payload={}
    files=[
    ('file',('download (90).png',open('/workspace/final_images/'+x,'rb'),'image/png'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.json())

