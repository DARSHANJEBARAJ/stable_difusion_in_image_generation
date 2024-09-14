import time
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
torch.set_default_dtype(torch.float16)
torch.backends.cudnn.benchmark = True 
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")
prompt = " flying cars and neon lights"
start_time = time.time()
with torch.no_grad():
    image = pipe(prompt).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
plt.imshow(image)
plt.axis("off") 
plt.show()
print(f"Time taken to generate the image: {elapsed_time:.2f} seconds")