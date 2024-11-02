from typing import Callable
import torch

def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = [
            image_processor(s.image).unsqueeze(0) for s in in_context_samples
        ]
        context_images = torch.cat(context_images, dim=0)
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images

def get_context_text(
        get_prompt: Callable[[dict], str],
        in_context_samples,
        num_shots,
        text_prompt='',
        instruction='',
    ) -> str:
        context_text = (
                "".join([get_prompt(s) for s in in_context_samples])
                if num_shots > 0
                else ""
            )
        
        context_text = text_prompt+context_text
        context_text = f"{instruction} {context_text}"
        return context_text