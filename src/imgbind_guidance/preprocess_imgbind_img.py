from torchvision import transforms
from torch import FloatTensor

data_transform = transforms.Compose(
  [
    transforms.Resize(
      224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.Normalize(
      mean=(0.48145466, 0.4578275, 0.40821073),
      std=(0.26862954, 0.26130258, 0.27577711),
    ),
  ]
)

def transform_vision_data(img: FloatTensor) -> FloatTensor:
  return data_transform(img)