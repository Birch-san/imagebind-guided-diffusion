from torch import Tensor, FloatTensor, BoolTensor
from diffusers.models import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from k_diffusion.external import DiscreteEpsDDPMDenoiser, DiscreteVDDPMDenoiser
from typing import Union, Optional, Protocol
import torch

class Denoiser(Protocol):
  sigma_min: FloatTensor
  sigma_max: FloatTensor
  sigmas: FloatTensor
  def forward(
    self,
    input: FloatTensor,
    sigma: FloatTensor,
    encoder_hidden_states: FloatTensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> FloatTensor: ...

class EPSDenoiser(DiscreteEpsDDPMDenoiser, Denoiser):
  inner_model: UNet2DConditionModel
  sampling_dtype: torch.dtype
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_eps(
    self,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: FloatTensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> FloatTensor:
    # encoder_attention_mask is a proposal from my cross_attn_3 branch of diffusers:
    # https://github.com/huggingface/diffusers/pull/2634
    # don't pass it in if we don't have to, to ensure compatibility with main branch of diffusers
    attn_kwargs = {} if cross_attention_mask is None else {
      'encoder_attention_mask': cross_attention_mask,
    }
    out: UNet2DConditionOutput = self.inner_model(
      sample.to(self.inner_model.dtype),
      timestep.to(self.inner_model.dtype),
      encoder_hidden_states=encoder_hidden_states.to(self.inner_model.dtype),
      return_dict=return_dict,
      **attn_kwargs,
    )
    return out.sample.to(self.sampling_dtype)

  def sigma_to_t(self, sigma: FloatTensor, quantize=None) -> FloatTensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)

class VDenoiser(DiscreteVDDPMDenoiser, Denoiser):
  sampling_dtype: torch.dtype
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_v(
    self,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: FloatTensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
    ) -> FloatTensor:
    # encoder_attention_mask is a proposal from my cross_attn_3 branch of diffusers:
    # https://github.com/huggingface/diffusers/pull/2634
    # don't pass it in if we don't have to, to ensure compatibility with main branch of diffusers
    attn_kwargs = {} if cross_attention_mask is None else {
      'encoder_attention_mask': cross_attention_mask,
    }
    out: UNet2DConditionOutput = self.inner_model(
      sample.to(self.inner_model.dtype),
      timestep.to(self.inner_model.dtype),
      encoder_hidden_states=encoder_hidden_states.to(self.inner_model.dtype),
      return_dict=return_dict,
      **attn_kwargs,
    )
    return out.sample.to(self.sampling_dtype)

  def sigma_to_t(self, sigma: FloatTensor, quantize=None) -> FloatTensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)
