import argparse
from inferencers import *
import logging
from transformers import AutoTokenizer, CLIPModel,AutoProcessor, AutoModelForVision2Seq
import sys
from open_flamingo_v2.open_flamingo.src.factory import create_model_and_transforms
from transformers import IdeficsForVisionText2Text, AutoProcessor
import json
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="open_flamingo_3b",# open_flamingo_9b;idefics_v2;open_flamingo_3b;idefics_v1
    )       
    parser.add_argument("--imagenet_root", type=str, default="/tmp")
    parser.add_argument("--dataset_mode", type=str, default="balanced") # balanced;imbalanced;
    parser.add_argument("--result_folder", type=str, default="./result")
    parser.add_argument("--method", type=str, default="Online_ICL")# FewShot;Online_ICL;
    parser.add_argument("--seed", type=int, default=42)     
    parser.add_argument("--stream", type=int, default=10000)    
    parser.add_argument("--bank", type=str, default="initial") #initial; total
    # Hyper parameters for OnlineICL
    parser.add_argument("--select_strategy", type=str, default="cosine")# cosine;l2;random
    parser.add_argument("--target_select", type=str, default="least_similarity") # prototype; random; least_similarity;most_similarity
    parser.add_argument("--dnum", type=int, default=4)
    parser.add_argument("--update_strategy", type=str, default="fixed") # noUpdate;gradient_prototype;cyclic;fixed;multi_step
    parser.add_argument("--M", type=int, default=1000) 
    parser.add_argument("--catergory_num", type=int, default=100) # 测100类 还是 1k类
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    device_set = "cuda:" + str(args.device)
    device = torch.device(device_set)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    if args.model == "open_flamingo_9b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="/data/share/mpt-7b/",
            tokenizer_path="/data/share/mpt-7b/",
            cross_attn_every_n_layers=4,
            precision="fp16",
            inference=True,
            device=device_set,
            checkpoint_path="/path/to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
        )
    elif args.model == "open_flamingo_3b":
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            precision="fp16",
            inference=True,
            device=device_set,
            checkpoint_path="/path/to/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.method == "Online_ICL":
        if args.model == "open_flamingo_9b":
            inferencer = Online_ICL(args, tokenizer, model, image_processor,device)
        elif args.model == "open_flamingo_3b":
            inferencer = Online_ICL(args, tokenizer, model, image_processor,device)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        results, predictions = inferencer.run()
    elif args.method == "FewShot":
        if args.model == "open_flamingo_9b":
            inferencer = FewShot(args, tokenizer, model, image_processor,  device)
        elif args.model == "open_flamingo_3b":
            inferencer = FewShot(args, tokenizer, model, image_processor, device)
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        results, predictions = inferencer.run()
    else:
        print("Method is invalid.")
        results = None

    results = {"device":args.device,"model": args.model,"dataset_mode":args.dataset_mode, "method": args.method, "select_strategy": args.select_strategy, "M": args.M,
               "batch_size":args.batch_size,"update_strategy":args.update_strategy,"alpha":args.alpha,"catergory_num":args.catergory_num,"dnum":args.dnum,"target_select":args.target_select,
               "stream":args.stream,"bank":args.bank,"results": results}
    print("-------------------------final-results-----------------------")
    print(results)
    
    if args.method == "Online_ICL": 
        res_file_last = os.path.join(args.result_folder, f"{args.model}-{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}-batch_size={args.batch_size}"
                                                f"-update_strategy={args.update_strategy}-alpha={args.alpha}-shot={args.dnum}-target_select={args.target_select}-stream={args.stream}.json")

        with open(res_file_last, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)
    else: # FewShot
        res_file = os.path.join(args.result_folder, f"{args.model}-{args.dataset_mode}-{args.method}-M={args.M}-select_strategy={args.select_strategy}-batch_size={args.batch_size}"
                                                f"-update_strategy={args.update_strategy}-shot={args.dnum}-bank={args.bank}.json")

        # load the prediction results to a json file
        with open(res_file, 'w') as json_file:
            json.dump(predictions, json_file, indent=4)

    