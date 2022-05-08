#!bin/bash
python3 check_cuda.py

python3 preprocess_span.py

python3 -u train_sharc.py \
    --train_batch=16 \
    --gradient_accumulation_steps=2 \
    --epoch=5 \
    --seed=115 \
    --learning_rate=5e-5 \
    --dsave="./out/{}" \
    --model=span \
    --early_stop=dev_0_combined \
    --data=./data/ \
    --data_type=span_roberta_base \
    --prefix=train_span \
    --eval_every_steps=100