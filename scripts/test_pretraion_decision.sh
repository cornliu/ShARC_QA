#!bin/bash
python3 train_sharc.py \
    --dsave="./out/{}" \
    --model=decision \
    --data=./data/ \
    --data_type=decision_roberta_base \
    --prefix=inference_decision \
    --resume=/work/jgtf0322/ShARC_QA/pretrained_models/decision.pt \
    --trans_layer=2 \
    --test \
    #--pretrained_lm_path="/work/jgtf0322/ShARC_QA/pretrained_models/roberta_base"