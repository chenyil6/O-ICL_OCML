from typing import List

from PIL import Image
import torch
import os

from openflamingo_v2.open_flamingo_chy.eval.eval_model import BaseEvalModel
from openflamingo_v2.open_flamingo_chy.src.factory import create_model_and_transforms
from contextlib import suppress
from openflamingo_v2.open_flamingo_chy.eval.models.utils import unwrap_model
from transformers import BatchFeature, IdeficsForVisionText2Text, IdeficsProcessor, AutoProcessor, AutoTokenizer  # 4.28.1 --> 4.36.2?

os.environ['HF_HOME'] = "/data/share/pyz/.cache/"
os.environ['TRANSFORMERS_CACHE'] = "/data/share/pyz/.cache/"

class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        assert (
            "checkpoint_path" in model_args  # hf_root
            and "precision" in model_args
        ), "IDEFICS requires..."

        self.device_num = model_args["device"]
        self.device = f"cuda:{self.device_num}" if model_args["device"] != "cpu" else "cpu"

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

        self.processor = AutoProcessor.from_pretrained(
            model_args["checkpoint_path"], local_files_only=True,cache_dir="/data/share/pyz/.cache")
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_args["checkpoint_path"],
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            cache_dir="/data/share/pyz/.cache"
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = 'left'
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id
        self.fake_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_prompt = self.fake_token + self.image_token + self.fake_token

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        inputs = self.processor(batch_text, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=True).input_ids

        with torch.inference_mode():
            with self.autocast():
                # outputs = unwrap_model(self.model).generate(
                #     self._prepare_images(batch_images).to(
                #         self.device, dtype=self.cast_dtype, non_blocking=True
                #     ),
                #     input_ids.to(self.device, dtype=torch.long, non_blocking=True),
                #     attention_mask=attention_mask.to(
                #         self.device, dtype=self.cast_dtype, non_blocking=True
                #     ),
                #     min_new_tokens=min_generation_length,
                #     max_new_tokens=max_generation_length,
                #     num_beams=num_beams,
                #     length_penalty=length_penalty,
                # )
                generated_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_generation_length,
                    eos_token_id=exit_condition,
                    bad_words_ids=bad_words_ids,
                    output_scores=True,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=2,
                    temperature=0.2,
                    top_k=20,
                    return_dict_in_generate=True
                )
                generated_ids = generated_outputs.sequences
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def get_logits(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
    ):
        with torch.inference_mode():
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=(past_key_values is not None),
                )
        return outputs

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None) -> str:
        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        icd_join_char = '\n'
        return f"{fake_token}{image_token}{fake_token}Question:{question} Short answer:{answer if answer is not None else ''}{icd_join_char if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    elif precision == "fp32":
        cast_dtype = torch.float32
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
