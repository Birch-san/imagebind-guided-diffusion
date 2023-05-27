# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import torch
from torch import Generator, compile
from PIL import Image
from typing import List, Callable
from os import makedirs, listdir
from os.path import join
import fnmatch
from pathlib import Path
from random import randint
import numpy as np
from numpy.typing import NDArray
import cv2
from controlnet_aux import HEDdetector

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

image: Image.Image = load_image('/home/birch/witch_512_512.png')
# image: NDArray = np.array(image)
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)

hed_image: Image.Image = hed(image, scribble=True)
# hed_image.save('out/witch_hed.png')

# controlnet: ControlNetModel = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
controlnet: ControlNetModel = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-scribble', torch_dtype=torch.float16)

vae: AutoencoderKL = AutoencoderKL.from_pretrained('hakurei/waifu-diffusion', subfolder='vae', torch_dtype=torch.float16)

# scheduler args documented here:
# https://github.com/huggingface/diffusers/blob/0392eceba8d42b24fcecc56b2cc1f4582dbefcc4/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L83
scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(
  'hakurei/waifu-diffusion',
#   'Birchlabs/wd-1-5-beta3-unofficial',
  subfolder='scheduler',
  # sde-dpmsolver++ is very new. if your diffusers version doesn't have it: use 'dpmsolver++' instead.
  algorithm_type='sde-dpmsolver++',
  # algorithm_type='dpmsolver++',
  solver_order=2,
  # solver_type='heun' seemed to give me black images. in k-diffusion it supposedly should give a sharper image. Cheng Lu reckons midpoint is better.
  solver_type='midpoint',
  use_karras_sigmas=True,
)

variant=None
# variant='ink'
# variant='mofu'
# variant='radiance'
# variant='illusion'
# pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
#   'hakurei/waifu-diffusion',
# #   'Birchlabs/wd-1-5-beta3-unofficial',
#   torch_dtype=torch.float16,
#   vae=vae,
#   scheduler=scheduler,
#   variant=variant,
#   revision='fp16'
# )
pipe: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_pretrained(
  'hakurei/waifu-diffusion',
  vae=vae,
  scheduler=scheduler,
  variant=variant,
  revision='fp16',
  controlnet=controlnet,
)
pipe.to('cuda')
compile(pipe.unet, mode='reduce-overhead')

sqrt_area=512

# WD1.5 was trained on area=896**2 and no side longer than 1152
# sqrt_area=896
# note: pipeline requires width and height to be multiples of 8
# height = 1024
# height = 896
height = 512
width = sqrt_area**2//height

# prompt = 'artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
prompt = 'kirisame marisa, carnelian, general content, broom_riding, full body, grin, looking at viewer, hair between eyes, floating hair, small breasts, touhou project, blonde hair, black dress, white ascot, puffy short sleeves, watercolor (medium), traditional media, painting (medium)'
# prompt = 'artoria pendragon (fate), reddizen, 1girl, best aesthetic, best quality, blue dress, full body, white shirt, blonde hair, looking at viewer, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, juliet sleeves, light smile, hair ribbon, outdoors, painting (medium), traditional media'
# negative_prompt = 'lowres, bad anatomy, bad hands, missing fingers, extra fingers, blurry, mutation, deformed face, ugly, bad proportions, monster, cropped, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic, realistic, real life, instagram'
negative_prompt = ''

uint32_iinfo = np.iinfo(np.uint32)
min, max = uint32_iinfo.min, uint32_iinfo.max

seed = randint(uint32_iinfo.min, uint32_iinfo.max)
# pipeline invocation args documented here:
# https://github.com/huggingface/diffusers/blob/0392eceba8d42b24fcecc56b2cc1f4582dbefcc4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#LL544C18-L544C18
out: StableDiffusionPipelineOutput = pipe.__call__(
  prompt,
  negative_prompt=negative_prompt,
  height=height,
  width=width,
  num_inference_steps=22,
  generator=Generator().manual_seed(seed),
  image=hed_image,
  num_images_per_prompt=8,
)
images: List[Image.Image] = out.images

out_dir = 'out_pipe'
makedirs(out_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0

for ix, img in enumerate(images):
  out_stem: str = f'{next_ix:05d}_{seed}'

  out_name: str = join(out_dir, f'{out_stem}.{ix}.jpg')
  img.save(out_name)
  print(f'Saved image: {out_name}')