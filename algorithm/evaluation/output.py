#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for generating pred-
# ictions on the training and test data:
# 
#  + init_dict:
#       - creates / resets dictionary to be used to output to results.csv
#  + get_header:
#       - used to write first row to results.csv
#  + prepare_file:
#       - begins process to write to results.csv
#  + write_row:
#       - using inputs from dictionary writes to results.csv
#  + close_file:
#       - closes results.csv 
#
# ###########################################################
#==============================================================================

import csv, os, datetime

def init_dict():
    output_dict = {}
    output_dict['run_by'] = 'Kristopher Lopez'
    output_dict['processor'] = 'Processor Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz, 3201 Mhz, 4 Cores'
    output_dict['memory'] = '16GB'
    output_dict['train_time'] = 0
    output_dict['train_accuracy'] = 0
    output_dict['train_precision'] = 0
    output_dict['train_recall'] = 0
    output_dict['train_fscore'] = 0
    output_dict['test_time'] = 0
    output_dict['test_accuracy'] = 0
    output_dict['test_precision'] = 0
    output_dict['test_recall'] = 0
    output_dict['test_fscore'] = 0
    output_dict['retained_variance'] = 1
    output_dict['preprocess_time'] = 0
    output_dict['total_time'] = 0
    output_dict['validation'] = 0
    return output_dict
                 
def add_metrics_to_dict(output_dict):
    return output_dict
    

def get_header():
    header = [
            'date_run', 
            'run_by', 
            'processor',
            'memory',
            'model',
            'observations',
            'dimensions',
            'retained_variance',
            'validation',
            'train_time',
            'train_accuracy',
            'train_precision',
            'train_recall',
            'train_fscore',
            'test_time',
            'test_accuracy',
            'test_precision',
            'test_recall',
            'test_fscore',
            'hype_a',
            'hype_c',
            'hype_d',
            'hype_e',
            'hype_k',
            'hype_n',
            'hype_loss',
            'hype_penalty',
            'preprocess_time',
            'total_time'
            ]
    return header

def prepare_file(output_directory, filename):

    filepath = os.path.join(output_directory, filename).replace('\\', '/')

    ofile  = open(filepath, 'w', newline='')
    ofile.truncate()
    output = csv.writer(ofile)
    output.writerow(get_header())
    
    return ofile, output

def write_row(wfile, params):
    row = []
    for v in get_header():
        if v == 'date_run':
            row.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            row.append(params.get(v))
    return wfile.writerow(row)
    
def close_file(ofile):
    ofile.close()

