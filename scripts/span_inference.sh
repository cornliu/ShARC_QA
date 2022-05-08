#!bin/bash
python3 check_cuda.py

# pip install transformers==2.8.0

python3 -u train_sharc.py \
--dsave="out/{}" \
--model=span \
--data=./data/ \
--data_type=span_roberta_base \
--prefix=inference_span \
--resume=./pretrained_models/span.pt \
--test \
--use_all_decisions=true