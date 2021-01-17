from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import csv
import os
from tqdm import tqdm

from collections import Counter

'''
Folder Structure
/VoxCeleb1/vox1_dev/id10001/1zcIwhmdeo4/00001.wav
/VoxCeleb1/vox1_test/id10270/5r0dWxy17C8/00001.wav
/VoxCeleb2/dev/aac/id00012/_raOc3-IRsw/00110.m4a
/VoxCeleb2/test/aac/id00017/_mjZ87sK6cA/00095.m4a
'''

def file_name_to_speaker_id(file_name):
    id_string = file_name.split('/')[-3]
    id_num = int(id_string.replace('id', ''))
    return id_num

def count_id_from_file_list(file_list):

    counter = Counter()

    for f in file_list:
        counter[file_name_to_speaker_id(f)] += 1

    return counter

def plot_counter_as_histogram(file_name, counter):
    plt.figure(figsize=(6, 6))

    distributions = np.array(list(counter.values()))

    plt.title(f'Speaker Counts (min. {min(distributions)})', fontsize=18)

    plt.hist(distributions, 
            bins=20,
            # bins=np.arange(-1, 1, 0.05), 
            alpha = 0.5)

    plt.savefig(file_name)
    plt.close()

    return

def get_mean_max_initial_acc(counter):

    distributions = np.array(list(counter.values()))

    mean_acc = 1 / len(distributions)

    max_acc = max(distributions) / sum(distributions)

    return mean_acc, max_acc

if __name__ ==  "__main__":
    vox1_dev_files = glob('VoxCeleb1/vox1_dev/*/*/*.wav')
    vox1_test_files = glob('VoxCeleb1/vox1_test/*/*/*.wav')
    vox2_dev_files = glob('VoxCeleb2/dev/aac/*/*/*.m4a')
    vox2_test_files = glob('VoxCeleb2/test/aac/*/*/*.m4a')

    print('{} {} {} {}'.format(*map(len, [vox1_dev_files, vox1_test_files, 
                                         vox2_dev_files, vox2_test_files])))

    vox1_dev_counter = count_id_from_file_list(vox1_dev_files)
    vox1_test_counter = count_id_from_file_list(vox1_test_files)
    vox2_dev_counter = count_id_from_file_list(vox2_dev_files)
    vox2_test_counter = count_id_from_file_list(vox2_test_files)

    plot_counter_as_histogram('./fig/vox1_dev.png', vox1_dev_counter)
    plot_counter_as_histogram('./fig/vox1_test.png', vox1_test_counter)
    plot_counter_as_histogram('./fig/vox2_dev.png', vox2_dev_counter)
    plot_counter_as_histogram('./fig/vox2_test.png', vox2_test_counter)

    print('Vox1 Dev  | Mean Acc. {:7.5f} | Biased max Acc. {:7.5f}'.format(
                *get_mean_max_initial_acc(vox1_dev_counter)))
    print('Vox1 Test | Mean Acc. {:7.5f} | Biased max Acc. {:7.5f}'.format(
                *get_mean_max_initial_acc(vox1_test_counter)))
    print('Vox2 Dev  | Mean Acc. {:7.5f} | Biased max Acc. {:7.5f}'.format(
                *get_mean_max_initial_acc(vox2_dev_counter)))
    print('Vox2 Test | Mean Acc. {:7.5f} | Biased max Acc. {:7.5f}'.format(
                *get_mean_max_initial_acc(vox2_test_counter)))

    # print(vox1_test_counter)