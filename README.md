# ShARC QA
Since we used the `pytorch0.4` this pytorch version is too old that is incompatible with docker, we write this readme. By following this readme you can train and test our work.

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
conda install pytorch==1.10.1 cudatoolkit=10.2 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2 transformers==2.8.0
pip install transformers==4.17.0
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

You can also download our pretrained models and our dev set predictions:
- Decision Making Model: [decision.pt](https://drive.google.com/file/d/1HZFi4p0tZtR5Z6msMVewi23rCdy3VclS/view?usp=sharing)
> We would now set up our directories like this:

```
.
└── model
    └── ...
└── segedu
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

## Discourse Segmentation of Rules (Section 2.1)

We use [SegBot](http://138.197.118.157:8000/segbot/) and [their implementation](https://www.dropbox.com/sh/tsr4ixfaosk2ecf/AACvXU6gbZfGLatPXDrzNcXCa?dl=0) to segment rules in the ShARC regulation snippets.

```shell
cd segedu
PYT_SEGBOT preprocess_discourse_segment.py
PYT_SEGBOT sharc_discourse_segmentation.py
```

`data/train_snippet_parsed.json` and `data/dev_snippet_parsed.json` are parsed rules.

## Fix Questions in ShARC

We find in some cases, there are some extra/missing spaces in ShARC questions. Here we fix them by merging these questions:

```shell
PYT_DISCERN fix_questions.py
```

## Using [splinter](https://arxiv.org/pdf/2101.00438.pdf) model to do decision Making (Section 2.2)
> preprocess: prepare inputs for splinter, generate labels for entailment supervision

```shell
PYT_DISCERN preprocess_decision.py
```

> training

```shell

PYT_DISCERN -u train_sharc.py \
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

> inference

During training, the checkpoint will be saved in `out/train_decision/`, you can use the checkpoint to do inference, for example

```shell
PYT_DISCERN train_sharc.py \
    --dsave="./out/{}" \
    --model=decision \
    --data=./data/ \
    --data_type=decision_splinter_base \
    --prefix=inference_decision \
    --resume=./out/train_decision/best.pt \
    --trans_layer=2 \
    --pretrained_lm_path="tau/splinter-base" \
    --test
```

