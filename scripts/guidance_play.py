from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoder_kl import AutoencoderKL
import torch
from torch import FloatTensor, Generator, randn
from os import makedirs, listdir
from os.path import join
from pathlib import Path
import fnmatch
from typing import List, Callable
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m_sde
from PIL import Image

from imgbind_guidance.schedule.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from imgbind_guidance.device import DeviceType, get_device_type
from imgbind_guidance.schedule.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from imgbind_guidance.clip_embed.embed_text_types import Embed, EmbeddingAndMask
from imgbind_guidance.clip_embed.embed_text import ClipImplementation, get_embedder
from imgbind_guidance.k_diff_wrappers.unet_2d_wrapper import EPSDenoiser, VDenoiser
from imgbind_guidance.latents_shape import LatentsShape
from imgbind_guidance.cfg_denoiser import CFGDenoiser
from imgbind_guidance.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from imgbind_guidance.log_intermediates import LogIntermediates, LogIntermediatesFactory, make_log_intermediates_factory

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

unet_dtype=torch.float16
vae_dtype=torch.float16
text_encoder_dtype=torch.float16
# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype=torch.float32

# WD1.5's Unet objective was parameterized on v-prediction
# https://twitter.com/RiversHaveWings/status/1578193039423852544
needs_vparam=True

# variant=None
# variant='ink'
# variant='mofu'
variant='radiance'
# variant='illusion'
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  'Birchlabs/wd-1-5-beta3-unofficial',
  torch_dtype=torch.float16,
  subfolder='unet',
  variant=variant,
).to(device).eval()
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  # WD1.5 uses WD1.4's VAE
  'hakurei/waifu-diffusion',
  subfolder='vae',
  torch_dtype=torch.float16,
).to(device).eval()
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)

embed: Embed = get_embedder(
  impl=ClipImplementation.HF,
  ckpt='Birchlabs/wd-1-5-beta3-unofficial',
  variant=variant,
  # WD1.5 is conditioned on penultimate hidden state of CLIP text encoder
  subtract_hidden_state_layers=1,
  # WD1.5 trained against concatenated CLIP segments, I think usually 3 of them?
  max_context_segments=3,
  device=device,
  torch_dtype=text_encoder_dtype,
)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = VDenoiser(unet, alphas_cumprod, sampling_dtype) if needs_vparam else EPSDenoiser(unet, alphas_cumprod, sampling_dtype)

schedule_template = KarrasScheduleTemplate.CudaMastering
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=unet_k_wrapped.sigma_min,
  model_sigma_max=unet_k_wrapped.sigma_max,
  device=unet_k_wrapped.sigmas.device,
  dtype=unet_k_wrapped.sigmas.dtype,
)

steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: FloatTensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max.cpu(),
  sigma_min=sigma_min.cpu(),
  rho=rho,
  device=device,
).to(sampling_dtype)

# WD1.5 was trained on area=896**2 and no side longer than 1152
sqrt_area=896
height = 1024
width = sqrt_area**2//height

latent_scale_factor = 1 << (len(vae.config.block_out_channels) - 1) # in other words, 8
latents_shape = LatentsShape(unet.in_channels, height // latent_scale_factor, width // latent_scale_factor)

seed = 1234
generator = Generator(device='cpu')

latents = randn((1, latents_shape.channels, latents_shape.height, latents_shape.width), dtype=sampling_dtype, device='cpu').to(device)
latents *= sigmas[0]

cond = '1girl, masterpiece, extremely detailed, light smile, outdoors, best quality, best aesthetic, floating hair, full body, ribbon, looking at viewer, hair between eyes, watercolor (medium), traditional media'
neg_cond = 'lowres, bad anatomy, bad hands, missing fingers, extra fingers, blurry, mutation, deformed face, ugly, bad proportions, monster, cropped, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic, realistic, real life, instagram'
conds = [neg_cond, cond]
embed_and_mask: EmbeddingAndMask = embed(conds)
embedding, _ = embed_and_mask

noise_sampler = BrownianTreeNoiseSampler(
  latents,
  # rather than using the sigma_{min,max} vars we already have:
  # refer to sigmas array, which can be truncated (e.g. if we were doing img2img)
  # we grab *penultimate* sigma_min, because final sigma is always 0
  sigma_min=sigmas[-2],
  sigma_max=sigmas[0],
  # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
  # I'm just re-using it because it's a convenient arbitrary number
  seed=seed,
)

denoiser = CFGDenoiser(unet_k_wrapped, cross_attention_conds=embedding)

denoised_latents: FloatTensor = sample_dpmpp_2m_sde(
  denoiser,
  latents,
  sigmas,
  noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers
  # callback=callback,
).to(vae_dtype)
del latents
pil_images: List[Image.Image] = latents_to_pils(denoised_latents)
del denoised_latents

out_dir = 'out'
makedirs(out_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0
out_stem: str = f'{next_ix:05d}_{seed}'


for stem, image in zip([out_stem], pil_images):
  out_name: str = join(out_dir, f'{stem}.png')
  image.save(out_name)
  print(f'Saved image: {out_name}')