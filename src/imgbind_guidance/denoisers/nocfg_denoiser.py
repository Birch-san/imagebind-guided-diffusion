from .unet_2d_wrapper import Denoiser
from torch import FloatTensor, BoolTensor
from dataclasses import dataclass
from typing import Optional

@dataclass
class NoCFGDenoiser:
  denoiser: Denoiser
  cross_attention_conds: FloatTensor
  cfg_scale: float = 7.5
  cross_attention_mask: Optional[BoolTensor] = None

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    return self.denoiser.forward(
      input=noised_latents,
      sigma=sigma,
      encoder_hidden_states=self.cross_attention_conds,
      cross_attention_mask=self.cross_attention_mask,
    )