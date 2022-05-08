#!bin/bash
python3 check_cuda.py
#python3 preprocess_span.py
python3 -u qg.py \
    --fin=./data/sharc_raw/json/sharc_dev.json \
    --fpred=./out/inference_span \
    --model_recover_path=/work/jgtf0322/ShARC_QA/pretrained_models/qg.bin \
    --cache_path=/work/jgtf0322/ShARC_QA/pretrained_models/unilm/