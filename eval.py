import pandas as pd
import numpy as np

# results = pd.read_csv ('./results/old/results_weighted_2.csv', sep=',')
# print('take average arcoss queries \n', results.mean())
# print('take average arcoss commits \n', results.groupby(['parent commit']).mean().mean())
# print('take average arcoss repos \n', results.groupby(['repo', 'parent commit']).mean().groupby(['repo']).mean().mean())


for level in range(12):
    read_path = './results/codebert/finetune/results_' + str(level+1) + '.csv'
    results = pd.read_csv (read_path, sep=',')
    print(level)
    print('take average arcoss queries \n', results.mean())
    print('take average arcoss commits \n', results.groupby(['parent commit']).mean().mean())
    print('take average arcoss repos \n', results.groupby(['repo', 'parent commit']).mean().groupby(['repo']).mean().mean())
    print('*' * 20)