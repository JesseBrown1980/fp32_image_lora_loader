# FP32 Image and LoRA Loader

This package provides classes for loading images in FP16 and FP32 formats and applying LoRA weights to PyTorch models. It is designed for optimized performance when working with large images and deep learning models.

## Features

- **FP16 Image Loader**: Loads images as FP16 tensors compatible with PyTorch.
- **FP32 Image Loader**: Loads images as FP32 tensors compatible with PyTorch.
- **LoRA Loader**: Loads LoRA weights and applies them to PyTorch models.

## Installation

To use this package, you need Python 3.7 or higher and pip installed on your system.

```bash
git clone https://github.com/JesseBrown1980/fp32_image_lora_loader.git
cd fp32_image_lora_loader
python setup.py install
Usage
Loading an FP16 Image
python

from fp32_image_lora_loader import FP16ImageLoader

image_loader = FP16ImageLoader()
tensor_image = image_loader.load_fp16_image("path/to/your/image.png")
Loading an FP32 Image
python

from fp32_image_lora_loader import FP32ImageLoader

image_loader = FP32ImageLoader()
tensor_image = image_loader.load_fp32_image("path/to/your/image.png")
Loading LoRA Weights
python

from fp32_image_lora_loader import LoRALoader

lora_loader = LoRALoader()
model = ...  # your PyTorch model
lora_loader.load_lora_weights(model, "path/to/lora_weights.safetensors")






## Usage Example


python

from fp32_image_lora_loader import FP32ImageLoader

# Initialize the image loader
image_loader = FP32ImageLoader()

# Load an image as an FP32 tensor
tensor_image = image_loader.load_fp32_image("path/to/your/image.png")

# tensor_image is now a PyTorch tensor in FP32 format