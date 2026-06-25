import pandas as pd
import numpy as np


def extract_answerABCDonly(s):
    if s == "CAN NOT ANSWER":
        return ''
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
    for choice in choices:
        if s.lower() in choice.lower():
            return choice[1]

    choices = ['A', 'B', 'C', 'D', 'E']
    for choice in choices:
        if s.lower() in choice.lower():
            return choice[0]
    return ''


def evaluate(data):

    cnt_rejected = 0
    data_un = data[~pd.isna(data['prediction'])]

    for idx in data['index']:
        ans = data.loc[data['index'] == idx, 'answer'].values[0]
        pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
        data.loc[data['index'] == idx, 'ori_prediction'] = data.loc[data['index'] == idx, 'prediction'].values[0]
        extract_pred = extract_answerABCDonly(pred)

        data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)
        data.loc[data['index'] == idx, 'prediction'] = extract_pred
        data.loc[data['index'] == idx, 'check'] = ""
    num_error = len(data) - len(data_un)
    num_rejected = cnt_rejected
    print(f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
          f'failed to obtain the score for {cnt_rejected} questions (prediction is not among options). '
          f'Those questions will be counted as 0 score in ALL rating.')

    #Calculate mean std across permutation per question
    scores = data['score'].values.reshape(-1, 4)
    std_per_question = scores.std(axis=1)
    mean_std = std_per_question.mean()
    print("Mean std across permutations per question:", mean_std)

    #Calculate std of accuracy per position
    std_acc_per_pos = data.groupby('answer')['score'].std() 
    mean_std_acc_per_pos = std_acc_per_pos.mean() 
    print("Mean std of accuracy per position:", mean_std_acc_per_pos)
    return data,num_error,num_rejected


def get_dimension_rating(data):
    TASKS = ['FP', 'PP', 'SR', 'DG']
    
    results = {}
    results['Overall'] = {}
    for task in TASKS:
        results[f'{task}'] = {}

    for i in range(len(data)):
        question = data.iloc[i]
        task = question['category']
        category = question['l2-category']
        if question['score'] >= 0:
            cnt = question['score']
            if category not in results[task].keys():
                results[task][f'{category}'] = {'true': int(cnt), 'false': int(1 - cnt)}
            else:
                results[task][f'{category}']['true'] += int(cnt)
                results[task][f'{category}']['false'] += int(1 - cnt)

    sum_all, succ_all = 0, 0
    for task, task_value in results.items():
        cnt_task, sum_task = 0, 0
        for category, category_dict in task_value.items():
            cnt_task += category_dict['true']
            sum_task += category_dict['false'] + category_dict['true']
            acc = category_dict['true'] / (category_dict['false'] + category_dict['true'])
            results[task][category]['Avg'] = acc
        if sum_task == 0:
            acc_tasks = 0
        else:
            acc_tasks = cnt_task / sum_task
        results[task]['Avg'] = acc_tasks
        results[task]['true'] = int(cnt_task)
        results[task]['false'] = int(sum_task - cnt_task)

        succ_all += cnt_task
        sum_all += sum_task
    results['Overall'] = succ_all / sum_all
    return results