# fp32_image_lora_loader/lora_loader.py

import torch
import os
import safetensors.torch

class LoRALoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_path": ("STRING", {"default": "path/to/lora_weights.safetensors"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_weights"
    CATEGORY = "ModelProcessing"
    TITLE = "Load LoRA Weights"

    def load_lora_weights(self, model, lora_path):
        # Validate LoRA weights path
        if not lora_path or not os.path.isfile(lora_path):
            print("Invalid LoRA weights path.")
            return model

        # Load LoRA weights using safetensors
        lora_weights = safetensors.torch.load_file(lora_path)

        # Apply LoRA weights to the model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in lora_weights:
                    param += lora_weights[name].to(param.device)

        return model

NODE_CLASS_MAPPINGS = {
    "LoRALoader": LoRALoader,
}
