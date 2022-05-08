#!bin/bash
python3 check_cuda.py
cd ./segedu
python3 preprocess_discourse_segment.py
python3 sharc_discourse_segmentation.py