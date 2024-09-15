import argparse
from inferencers import *
import logging
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.append('/data/chy/online')
from open_flamingo_4.open_flamingo.src.factory import create_model_and_transforms
from transformers import IdeficsForVisionText2Text, AutoProcessor
import json

logging.getLogger("transformers").setLevel(logging.CRITICAL)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="open_flamingo",
    )       

    parser.add_argument("--imagenet_root", type=str, default="/tmp")
    parser.add_argument("--dataset_mode", type=str, default="unbalanced")
    parser.add_argument("--result_folder", type=str, default="./result")
    parser.add_argument("--method", type=str, default="FewShot")# FewShot;
    parser.add_argument("--seed", type=int, default=42)     
    # Hyper parameters for DAIL
    parser.add_argument("--select_strategy", type=str, default="topk")
    parser.add_argument("--update_strategy", type=str, default="noUpdate") # 
    parser.add_argument("--M", type=int, default=500)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    device_set = "cuda:" + str(args.device)
    device = torch.device(device_set)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    if args.model == "open_flamingo":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="/data/share/mpt-7b/",
            tokenizer_path="/data/share/mpt-7b/",
            cross_attn_every_n_layers=4,
            precision="fp16",
            inference=True,
            device=device_set,
            checkpoint_path="/data/share/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
        )
    elif args.model == "idefics":
        checkpoint = "/data/share/pyz/model_weight/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(checkpoint,local_files_only=True, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(checkpoint)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # embedding model
    embedding_model = CLIPModel.from_pretrained('/home/chy63/.cache/huggingface/hub/models--clip-vit-base-patch32',local_files_only=True)
    embedding_processor = CLIPProcessor.from_pretrained(
            '/home/chy63/.cache/huggingface/hub/models--clip-vit-base-patch32',local_files_only=True)
    print("load clip successfully...")

    if args.method == "Online_ICL":
        if args.model == "open_flamingo":
            inferencer = Online_ICL(args, tokenizer, model, image_processor, embedding_model, embedding_processor, device)
        elif args.model == "idefics":
            inferencer = Online_ICL(args, tokenizer, model, image_processor, embedding_model, embedding_processor, device,processor=processor)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        results, predictions = inferencer.run()

    elif args.method == "FewShot":
        if args.model == "open_flamingo":
            inferencer = FewShot(args, tokenizer, model, image_processor, embedding_model, embedding_processor, device)
        elif args.model == "idefics":
            inferencer = FewShot(args, tokenizer, model, image_processor, embedding_model, embedding_processor, device,processor=processor)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        results, predictions = inferencer.run()
    else:
        print("Method is invalid.")
        results = None

    if args.method == "Online_ICL":
        res_file_online = os.path.join(args.result_folder, f"online-{args.model}-{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}"
                                                f"-update_strategy={args.update_strategy}-.json")
        
        res_file_last = os.path.join(args.result_folder, f"last-{args.model}-{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}"
                                                f"-update_strategy={args.update_strategy}-.json")

        # load the prediction results to a json file
        with open(res_file_online, 'w') as json_file:
            json.dump(predictions["0"], json_file, indent=4)
        
        with open(res_file_online, 'w') as json_file:
            json.dump(predictions["1"], json_file, indent=4)
    else: # FewShot
        res_file = os.path.join(args.result_folder, f"last-{args.model}-{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}"
                                                f"-update_strategy={args.update_strategy}-.json")

        # load the prediction results to a json file
        with open(res_file, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)

    results = {"model": args.model,"dataset_mode":args.dataset_mode, "method": args.method, "select_strategy": args.select_strategy, "M": args.M,
               "update_strategy": args.update_strategy,"results": results}
    print("-------------------------final-results-----------------------")
    print(results)