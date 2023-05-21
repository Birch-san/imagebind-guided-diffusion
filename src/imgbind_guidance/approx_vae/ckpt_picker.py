from .decoder_ckpt import DecoderCkpt
from .encoder_ckpt import EncoderCkpt

def get_approx_decoder_ckpt(model_name: str, wd_prefer_1_3 = True) -> DecoderCkpt:
  """
  model_name was meant to indicate which UNet you're using, but now that I think about it, latent->RGB
  conversion is the job of the VAE. So probably the decoder I trained on WD1.4's outputs has similar weights to WD1.5's.
  """
  match model_name:
    case 'CompVis/stable-diffusion-v1-4':
      return DecoderCkpt.SD1_4
    case 'hakurei/waifu-diffusion':
      return DecoderCkpt.WD1_3 if wd_prefer_1_3 else DecoderCkpt.WD1_4
    case 'waifu-diffusion/wd-1-5-beta2' | 'waifu-diffusion/wd-1-5-beta3':
      return DecoderCkpt.WD1_5
    case 'runwayml/stable-diffusion-v1-5' | _:
      return DecoderCkpt.SD1_5

def get_approx_encoder_ckpt(model_name: str, wd_prefer_1_3 = True) -> EncoderCkpt:
  """
  model_name was meant to indicate which UNet you're using, but now that I think about it, RGB->latent
  conversion is the job of the VAE. So probably the encoder I trained on WD1.4's outputs has similar weights to WD1.5's.
  """
  match model_name:
    case 'CompVis/stable-diffusion-v1-4':
      return EncoderCkpt.SD1_4
    case 'hakurei/waifu-diffusion':
      return EncoderCkpt.WD1_3 if wd_prefer_1_3 else EncoderCkpt.WD1_4
    case 'waifu-diffusion/wd-1-5-beta2' | 'waifu-diffusion/wd-1-5-beta3':
      return EncoderCkpt.WD1_5
    case 'runwayml/stable-diffusion-v1-5' | _:
      return EncoderCkpt.SD1_5