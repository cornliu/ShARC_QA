# ShARC QA
This folder contains several scripts that help you to train or test our model.

## Requirements
> Discourse segmentation environment (`PYT_SEGBOT`)

```bash
conda create -n segbot python=3.6
conda install pytorch==0.4.1 -c pytorch
conda install nltk==3.4.5 numpy==1.18.1 pycparser==2.20 six==1.14.0 tqdm==4.44.1
```

> Main environment (`PYT_DISCERN`)

```bash
conda create -n discern python=3.6
conda install pytorch==1.0.1 cudatoolkit=10.0 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2 transformers==2.8.0
```

> UniLM question generation environment (`PYT_QG`)

```bash
# create conda environment
conda create -n qg python=3.6
conda install pytorch==1.1 cudatoolkit=10.0 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2

# install apex
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --cuda_ext --cpp_ext
cd ..

# setup unilm
cd qg
pip install --editable .
```

> Download ShARC data
```bash
mkdir data
cd data
wget https://sharc-data.github.io/data/sharc1-official.zip -O sharc_raw.zip
unzip sharc_raw.zip
mv sharc1-official/ sharc_raw
cd ..
```

> Download RoBERTa, UniLM
```bash
mkdir pretrained_models
# RoBERTa
mkdir pretrained_models/roberta_base
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json -O pretrained_models/roberta_base/config.json
wget --quiet https://cdn.huggingface.co/roberta-base-merges.txt -O pretrained_models/roberta_base/merges.txt
wget --quiet https://cdn.huggingface.co/roberta-base-pytorch_model.bin -O pretrained_models/roberta_base/pytorch_model.bin
wget --quiet https://cdn.huggingface.co/roberta-base-vocab.json -O pretrained_models/roberta_base/vocab.json
# UniLM & BERT
mkdir pretrained_models/unilm
wget --quiet https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin -O pretrained_models/unilm/unilmv1-large-cased.bin
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt -O pretrained_models/unilm/bert-large-cased-vocab.txt
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz -O pretrained_models/unilm/bert-large-cased.tar.gz
cd pretrained_models/unilm
tar -zxvf bert-large-cased.tar.gz
rm bert-large-cased.tar.gz
```
You can also download our pretrained models and our dev set predictions:
- Decision Making Model: [decision.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/ESrX-zVkGjZBsHofDoChXIYB1UVlgld3jJZyeJcZApemCQ?e=ggQy1w)
- Span Extraction Model: [span.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/EZ7e3LDwYZlHtg917fTsK2gBUp7HD52wzE65mKSx4FY5uQ?e=kVXjyC)
- Question Generation Model: [unilmqg.bin](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/ER4GRoby0ORGjIYUKLo-Tc4Bq-G7De5qElmJh_Rt_EtGqQ?e=8Yeuix)
> We would now set up our directories like this:

```
.
└── model
    └── ...
└── segedu
    └── ...
└── unilmqg
    └── ...
└── README.md
└── data
    └── ...
└── pretrained_models
    └── unilm
        └── ...
        └── unilmqg.bin
    └── roberta_base
        └── ...
    └── decision.pt
    └── span.pt
```

## Using [splinter](https://arxiv.org/pdf/2101.00438.pdf) model to do decision Making (Section 2.2)
> Update transformers version to 4.17.0 under PYT_DISCERN environment
```shell
pip install transformers==4.17.0
```
> preprocess: prepare inputs for RoBERTa, generate labels for entailment supervision

```shell
PYT_DISCERN preprocess_decision.py
```

> training

```shell
MODEL="splinter"
python3 -u train_sharc.py \
    --train_batch=16 \
    --gradient_accumulation_steps=2 \
    --epoch=5 \
    --seed=323 \
    --learning_rate=5e-5 \
    --loss_entail_weight=3.0 \
    --dsave="./out/{}" \
    --model=decision \
    --early_stop=dev_0a_combined \
    --data=./data/ \
    --data_type=decision_splinter_base \
    --prefix=train_decision \
    --trans_layer=2 \
    --eval_every_steps=300 \
    --pretrained_lm_path="tau/splinter-base"
```



