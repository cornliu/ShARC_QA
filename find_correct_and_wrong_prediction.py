import json
from argparse import ArgumentParser
import os
import sys

parser = ArgumentParser()
parser.add_argument('--prediction_file', default="./out/train_decision_no_rule_change_VAT2loan/dev.preds.json", help='prediction file directory')
parser.add_argument('--ground_truth_file', default="./data/sharc_raw/json/sharc_dev_no_rule_change_VAT2loan_question_fixed.json", help='ground truth file directory')
parser.add_argument('--find_correct_prediction_file', default="./find_prediction_file/correct_prediction_wo_rule_no_rule_change_VAT2loan.json")
parser.add_argument('--find_wrong_prediction_file', default="./find_prediction_file/wrong_prediction_wo_rule_no_rule_change_VAT2loan.json")
args = parser.parse_args()

prediction_file = json.load(open(args.prediction_file))
ground_truth_file = json.load(open(args.ground_truth_file))
assert len(prediction_file) == len(ground_truth_file)

print({'utterance_id': ground_truth_file[0]['utterance_id'],
'pred_answer': prediction_file[0]['pred_answer'].lower(),
'answer': ground_truth_file[0]['answer'].lower(),
'snippet': ground_truth_file[0]['snippet'],
'question': ground_truth_file[0]['question'],
'scenario': ground_truth_file[0]['scenario'],
'history': ground_truth_file[0]['history'],
'evidence': ground_truth_file[0]['evidence']
})
# sys.exit()

correct_prediction = []
wrong_prediction = []

for data_index in range(len(prediction_file)):
    if ground_truth_file[data_index]['answer'].lower() not in ['yes', 'no', 'irrelevant']:
        answer = 'more'
    else:
        answer = ground_truth_file[data_index]['answer'].lower()

    data_temp = {'utterance_id': ground_truth_file[data_index]['utterance_id'],
                'pred_answer': prediction_file[data_index]['pred_answer'].lower(),
                'answer': answer,
                'snippet': ground_truth_file[data_index]['snippet'],
                'question': ground_truth_file[data_index]['question'],
                'scenario': ground_truth_file[data_index]['scenario'],
                'history': ground_truth_file[data_index]['history'],
                'evidence': ground_truth_file[data_index]['evidence']
                }

    if prediction_file[data_index]['pred_answer'].lower() == answer:
        print("correct {}".format(answer))
        correct_prediction.append(data_temp)
    else:
        print("pred: {}, gt: {}".format(prediction_file[data_index]['pred_answer'].lower(), answer))
        wrong_prediction.append(data_temp)

# print(correct_prediction)
# print(wrong_prediction)
assert len(prediction_file) == (len(correct_prediction) + len(wrong_prediction))

with open(args.find_correct_prediction_file, 'wt') as f:
    json.dump(correct_prediction, f, indent=2)

with open(args.find_wrong_prediction_file, 'wt') as f:
    json.dump(wrong_prediction, f, indent=2)