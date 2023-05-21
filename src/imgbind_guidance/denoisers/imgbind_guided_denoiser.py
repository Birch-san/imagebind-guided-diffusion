from .unet_2d_wrapper import Denoiser
from torch import FloatTensor, BoolTensor, no_grad, enable_grad
from torch import autograd
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from models.imagebind_model import ImageBindModel, ModalityType
from data import load_and_transform_vision_data
from imgbind_guidance.approx_vae.latent_roundtrip import LatentsToRGB, RGBToLatents
from ..preprocess_imgbind_img import transform_vision_data

def spherical_dist_loss(x: FloatTensor, y: FloatTensor) -> FloatTensor:
  x = F.normalize(x, dim=-1)
  y = F.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

@dataclass
class ImgBindGuidedCFGDenoiser:
  denoiser: Denoiser
  imgbind: ImageBindModel
  latents_to_rgb: LatentsToRGB
  rgb_to_latents: RGBToLatents
  cross_attention_conds: FloatTensor
  target_imgbind_cond: FloatTensor
  cfg_scale: float = 7.5
  guidance_scale: float = 50.
  cross_attention_mask: Optional[BoolTensor] = None

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    noised_latents = noised_latents.detach().requires_grad_()
    with enable_grad():
      denoised: FloatTensor = self.denoiser.forward(
        input=noised_latents,
        sigma=sigma,
        encoder_hidden_states=self.cross_attention_conds,
        cross_attention_mask=self.cross_attention_mask,
      )
      decoded: FloatTensor = self.latents_to_rgb(denoised)
      preprocessed: FloatTensor = transform_vision_data(decoded)
      imgbind_inputs: Dict[ModalityType, FloatTensor] = {
        ModalityType.VISION: preprocessed,
      }
      imgbind_outputs: Dict[ModalityType, FloatTensor] = self.imgbind(imgbind_inputs)
      imgbind_output: FloatTensor = imgbind_outputs[ModalityType.VISION][0]
      loss: FloatTensor = spherical_dist_loss(imgbind_output, self.target_imgbind_cond).sum() * self.guidance_scale
      vec_jacobians: Tuple[FloatTensor, ...] = autograd.grad(loss, noised_latents)
      grad: FloatTensor = -vec_jacobians[0].detach()
    guided_cond: FloatTensor = denoised.detach() + grad * sigma**2
    return guided_cond