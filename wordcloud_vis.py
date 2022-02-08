# -*- coding: utf-8 -*-
import nltk
from wordcloud import WordCloud
import json
from argparse import ArgumentParser
import os
import sys
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--prediction_file', default="./find_prediction_file/wrong_prediction_wo_rule_change_VAT2loan.json")
args = parser.parse_args()

prediction_file = json.load(open(args.prediction_file))

if 'wo' in args.prediction_file:
    pred_yes_list = []
    pred_no_list = []
    pred_more_list = []
    pred_irrelevant_list = []
    for data_idx in range(len(prediction_file)): # get all the question
        if prediction_file[data_idx]['pred_answer'] == 'yes':
            pred_yes_list.append(prediction_file[data_idx]['question'])
        elif prediction_file[data_idx]['pred_answer'] == 'no':
            pred_no_list.append(prediction_file[data_idx]['question'])
        elif prediction_file[data_idx]['pred_answer'] == 'more':
            pred_more_list.append(prediction_file[data_idx]['question'])
        else: # irrelevant
            pred_irrelevant_list.append(prediction_file[data_idx]['question'])
    
    for pred in ['yes', 'no', 'more', 'irrelevant']:
        if pred == 'yes':
            pred_list = pred_yes_list
        elif pred == 'no':
            pred_list = pred_no_list
        elif pred == 'more':
            pred_list = pred_more_list
        else: # irrelevant
            pred_list = pred_irrelevant_list
        pred_list = ' '.join(pred_list)
        count_result = pd.value_counts(nltk.word_tokenize(pred_list))
        nltk_question_list = ' '.join(nltk.word_tokenize(pred_list))
        cloud = WordCloud().generate(nltk_question_list)
        cloud.to_file('{}_pred_{}.png'.format(args.prediction_file[:-5], pred))
        count_result.to_csv('{}_pred_{}.csv'.format(args.prediction_file[:-5], pred))
else:
    pass