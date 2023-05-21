from .unet_2d_wrapper import Denoiser
from torch import FloatTensor, BoolTensor, enable_grad
from torch import autograd
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from models.imagebind_model import ImageBindModel, ModalityType
from imgbind_guidance.approx_vae.latent_roundtrip import LatentsToRGB
from ..preprocess_imgbind_img import transform_vision_data
from ..spherical_dist_loss import spherical_dist_loss

@dataclass
class ImgBindGuidedCFGDenoiser:
  denoiser: Denoiser
  imgbind: ImageBindModel
  latents_to_rgb: LatentsToRGB
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
    uncond_noised_latents = noised_latents.detach()
    cond_noised_latents = noised_latents.detach().requires_grad_()
    del noised_latents
    uncond_denoised: FloatTensor = self.denoiser.forward(
      input=uncond_noised_latents,
      sigma=sigma,
      encoder_hidden_states=self.cross_attention_conds[:1],
      cross_attention_mask=None if self.cross_attention_mask is None else self.cross_attention_mask[:1],
    )
    del uncond_noised_latents
    with enable_grad():
      cond_denoised: FloatTensor = self.denoiser.forward(
        input=cond_noised_latents,
        sigma=sigma,
        encoder_hidden_states=self.cross_attention_conds[1:],
        cross_attention_mask=None if self.cross_attention_mask is None else self.cross_attention_mask[1:],
      )
      decoded: FloatTensor = self.latents_to_rgb(cond_denoised)
      preprocessed: FloatTensor = transform_vision_data(decoded)
      imgbind_inputs: Dict[ModalityType, FloatTensor] = {
        ModalityType.VISION: preprocessed,
      }
      imgbind_outputs: Dict[ModalityType, FloatTensor] = self.imgbind(imgbind_inputs)
      imgbind_output: FloatTensor = imgbind_outputs[ModalityType.VISION][0]
      loss: FloatTensor = spherical_dist_loss(imgbind_output, self.target_imgbind_cond).sum() * self.guidance_scale
      vec_jacobians: Tuple[FloatTensor, ...] = autograd.grad(loss, cond_noised_latents)
      grad: FloatTensor = -vec_jacobians[0].detach()
    guided_cond: FloatTensor = cond_denoised.detach() + grad * sigma**2
    return uncond_denoised + (guided_cond - uncond_denoised) * self.cfg_scale