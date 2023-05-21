from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoder_kl import AutoencoderKL
import torch
from torch import FloatTensor, Generator, randn
from os import makedirs, listdir
from os.path import join
from pathlib import Path
import fnmatch
from typing import List, Callable, Dict
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m_sde, sample_dpmpp_2m
from PIL import Image
from models.imagebind_model import imagebind_huge, ImageBindModel, ModalityType
import data
from data import load_and_transform_text#, load_and_transform_vision_data, load_and_transform_audio_data

from imgbind_guidance.schedule.schedule_params import get_alphas, get_alphas_cumprod, get_betas
from imgbind_guidance.device import DeviceType, get_device_type
from imgbind_guidance.schedule.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from imgbind_guidance.clip_embed.embed_text_types import Embed, EmbeddingAndMask
from imgbind_guidance.clip_embed.embed_text import ClipImplementation, get_embedder
from imgbind_guidance.denoisers.unet_2d_wrapper import EPSDenoiser, VDenoiser
from imgbind_guidance.latents_shape import LatentsShape
from imgbind_guidance.denoisers.imgbind_guided_nocfg_denoiser import ImgBindGuidedNoCFGDenoiser
from imgbind_guidance.denoisers.imgbind_guided_cfg_denoiser import ImgBindGuidedCFGDenoiser
from imgbind_guidance.denoisers.nocfg_denoiser import NoCFGDenoiser
from imgbind_guidance.denoisers.cfg_denoiser import CFGDenoiser
from imgbind_guidance.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from imgbind_guidance.log_intermediates import LogIntermediates, LogIntermediatesFactory, make_log_intermediates_factory
from imgbind_guidance.approx_vae.latents_to_pils import make_approx_latents_to_pils
from imgbind_guidance.approx_vae.decoder_ckpt import DecoderCkpt
from imgbind_guidance.approx_vae.encoder_ckpt import EncoderCkpt
from imgbind_guidance.approx_vae.decoder import Decoder
from imgbind_guidance.approx_vae.encoder import Encoder
from imgbind_guidance.approx_vae.get_approx_decoder import get_approx_decoder
from imgbind_guidance.approx_vae.get_approx_encoder import get_approx_encoder
from imgbind_guidance.approx_vae.latent_roundtrip import LatentsToRGB, RGBToLatents, make_approx_latents_to_rgb, make_approx_rgb_to_latents, make_real_latents_to_rgb, make_real_rgb_to_latents
from imgbind_guidance.approx_vae.ckpt_picker import get_approx_decoder_ckpt, get_approx_encoder_ckpt

# relative to current working directory, i.e. repository root of embedding-compare
img_bind_dir = 'lib/ImageBind'

data.BPE_PATH = join(img_bind_dir, data.BPE_PATH)
img_bind_assets_dir = join(img_bind_dir, '.assets')
assets_dir = 'assets'

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
).to(device).eval().requires_grad_(False)
torch.compile(unet, mode='reduce-overhead')
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  # WD1.5 uses WD1.4's VAE
  'hakurei/waifu-diffusion',
  subfolder='vae',
  torch_dtype=torch.float16,
).to(device).eval().requires_grad_(False)
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)

guidance_use_approx_vae = False
approx_decoder_ckpt: DecoderCkpt = get_approx_decoder_ckpt('waifu-diffusion/wd-1-5-beta3')
approx_decoder: Decoder = get_approx_decoder(approx_decoder_ckpt, device)
approx_encoder_ckpt: EncoderCkpt = get_approx_encoder_ckpt('waifu-diffusion/wd-1-5-beta3')
approx_encoder: Encoder = get_approx_encoder(approx_encoder_ckpt, device)
approx_latents_to_pils: LatentsToPils = make_approx_latents_to_pils(approx_decoder)
approx_guidance_decoder: LatentsToRGB = make_approx_latents_to_rgb(approx_decoder)
approx_guidance_encoder: RGBToLatents = make_approx_rgb_to_latents(approx_encoder)
real_guidance_decoder: LatentsToRGB = make_real_latents_to_rgb(vae)
real_guidance_encoder: RGBToLatents = make_real_rgb_to_latents(vae)
guidance_decoder: LatentsToRGB = approx_guidance_decoder if guidance_use_approx_vae else real_guidance_decoder
guidance_encoder: RGBToLatents = approx_guidance_encoder if guidance_use_approx_vae else real_guidance_encoder

log_intermediates_approx_decode = True
intermediate_latents_to_pils: LatentsToPils = approx_latents_to_pils if log_intermediates_approx_decode else latents_to_pils
make_log_intermediates: LogIntermediatesFactory = make_log_intermediates_factory(intermediate_latents_to_pils)
log_intermediates_enabled = False

if log_intermediates_enabled and not log_intermediates_approx_decode or not guidance_use_approx_vae:
  # if we're expecting to invoke VAE every sampler step: make it cheaper to do so
  torch.compile(vae, mode='reduce-overhead')

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

schedule_template = KarrasScheduleTemplate.Mastering
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
# height = 1024
height = 896
width = sqrt_area**2//height

latent_scale_factor = 1 << (len(vae.config.block_out_channels) - 1) # in other words, 8
latents_shape = LatentsShape(unet.in_channels, height // latent_scale_factor, width // latent_scale_factor)

seed = 1234
generator = Generator(device='cpu')

latents = randn((1, latents_shape.channels, latents_shape.height, latents_shape.width), dtype=sampling_dtype, device='cpu', generator=generator).to(device)
latents *= sigmas[0]

cond = '1girl, masterpiece, extremely detailed, light smile, outdoors, best quality, best aesthetic, floating hair, full body, ribbon, looking at viewer, hair between eyes, watercolor (medium), traditional media'
neg_cond = 'lowres, bad anatomy, bad hands, missing fingers, extra fingers, blurry, mutation, deformed face, ugly, bad proportions, monster, cropped, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic, realistic, real life, instagram'
conds = [neg_cond, cond]
# conds = [cond]
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

imgbind: ImageBindModel = imagebind_huge(pretrained=True).to(device).eval().requires_grad_(False)
torch.compile(imgbind, mode='reduce-overhead')
text_bind=['watercolour illustration of a girl sitting by a campfire at night']
image_bind_path=[join(img_bind_assets_dir, asset) for asset in []]
audio_bind_path=[join(img_bind_assets_dir, asset) for asset in []]
imgbind_inputs: Dict[ModalityType, FloatTensor] = {
  ModalityType.TEXT: load_and_transform_text(text_bind, device),
  # ModalityType.VISION: load_and_transform_vision_data(image_bind_path, device),
  # ModalityType.AUDIO: load_and_transform_audio_data(audio_bind_path, device),
}
imgbind_out: Dict[ModalityType, FloatTensor] = imgbind.forward(imgbind_inputs)
target_imgbind_cond: FloatTensor = imgbind_out[ModalityType.TEXT][0]
guidance_scale=300.
cfg_scale=2.
denoiser = ImgBindGuidedCFGDenoiser(
  denoiser=unet_k_wrapped,
  imgbind=imgbind,
  latents_to_rgb=guidance_decoder,
  target_imgbind_cond=target_imgbind_cond,
  cross_attention_conds=embedding,
  guidance_scale=guidance_scale,
  cfg_scale=cfg_scale,
)
# denoiser = ImgBindGuidedNoCFGDenoiser(
#   denoiser=unet_k_wrapped,
#   imgbind=imgbind,
#   latents_to_rgb=guidance_decoder,
#   target_imgbind_cond=target_imgbind_cond,
#   cross_attention_conds=embedding,
#   guidance_scale=guidance_scale,
# )
# denoiser = CFGDenoiser(
#   denoiser=unet_k_wrapped,
#   cross_attention_conds=embedding,
#   cfg_scale=7.5,
# )
# denoiser = NoCFGDenoiser(
#   denoiser=unet_k_wrapped,
#   cross_attention_conds=embedding,
# )

out_dir = 'out'
makedirs(out_dir, exist_ok=True)
intermediates_dir=join(out_dir, 'intermediates')
makedirs(intermediates_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0
out_stem: str = f'{next_ix:05d}_{seed}_{guidance_scale}_{cfg_scale}'

if log_intermediates_enabled:
  intermediates_path = join(intermediates_dir, out_stem)
  makedirs(intermediates_path, exist_ok=True)
  callback: LogIntermediates = make_log_intermediates([intermediates_path])
else:
  callback = None

denoised_latents: FloatTensor = sample_dpmpp_2m(
  denoiser,
  latents,
  sigmas,
  # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers
  callback=callback,
).to(vae_dtype)
del latents
pil_images: List[Image.Image] = latents_to_pils(denoised_latents)
del denoised_latents


for stem, image in zip([out_stem], pil_images):
  # TODO: put this back to png once we've stopped prototyping
  out_name: str = join(out_dir, f'{stem}.jpg')
  image.save(out_name)
  print(f'Saved image: {out_name}')