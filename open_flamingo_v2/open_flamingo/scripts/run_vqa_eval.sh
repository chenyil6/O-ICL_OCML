DEVICE=0 # gpu num

RANDOM_ID="VQAV2_RS"
RESULTS_FILE="/data/chy/open_flamingo/results/${RANDOM_ID}.json"

nohup python -u open_flamingo/eval/evaluate_siir.py \
    --model "open_flamingo" \
    --lm_path "/data/share/mpt-7b/" \
    --lm_tokenizer_path "/data/share/mpt-7b/" \
    --checkpoint_path "/data/share/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vqav2_train_image_dir_path "/data/wyl/data/coco/train2014/"  \
    --vqav2_train_questions_json_path "/data/share/pyz/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_train_annotations_json_path  "/data/share/pyz/data/vqav2/v2_mscoco_train2014_annotations.json" \
    --vqav2_test_image_dir_path "/data/wyl/data/coco/val2014/" \
    --vqav2_test_questions_json_path "/data/share/pyz/data/vqav2/v2_mscoco_val2014_question_subdata.json" \
    --vqav2_test_annotations_json_path "/data/share/pyz/data/vqav2/v2_mscoco_val2014_annotations_subdata.json" \
    --results_file $RESULTS_FILE \
    --num_samples 5 \
    --shots 2 \
    --num_trials 1 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --eval_vqav2  > RECORD_VQAV2_RS.log 2>&1 &


echo "evaluation complete! results written to ${RESULTS_FILE}"