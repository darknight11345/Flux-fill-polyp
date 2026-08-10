import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

image = load_image("/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/guided_diffusion/segmented-images/images/147257f5-7401-4f3a-bacb-7c16611c2ef1.jpg")
mask = load_image("/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/guided_diffusion/segmented-images/masks/147257f5-7401-4f3a-bacb-7c16611c2ef1.jpg")

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"flux-fill-dev.png")
