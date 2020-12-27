from glob import glob
import json
import sys
from mel2samp_tacotron2 import Mel2SampWaveglow
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import librosa
import csv
import os

from functools import partial

from tqdm import tqdm

'''
Folder Structure
/VoxCeleb1/vox1_dev/id10001/1zcIwhmdeo4/00001.wav
/VoxCeleb1/vox1_test/id10270/5r0dWxy17C8/00001.wav
/VoxCeleb2/dev/aac/id00012/_raOc3-IRsw/00110.m4a
/VoxCeleb2/test/aac/id00017/_mjZ87sK6cA/00095.m4a
'''

CONFIGURATION_FILE = 'config.json'

T_THRES = 19

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    mel_config = json_info["mel_config"]
    MEL2SAMPWAVEGLOW = Mel2SampWaveglow(**mel_config)

    hp = json_info["hp"]

    global_scope = sys.modules[__name__]

    for key in hp:
        setattr(global_scope, key, hp[key])
        print(f'{key} == {hp[key]}')

def struct_meta(file_list, mode='vox1'):
    if mode == 'vox1':
        meta = [(file, file.split('/')[2], 
                librosa.get_duration(filename=file), 
                librosa.get_samplerate(file)
                ) for file in tqdm(file_list)]
    elif  mode == 'vox2':
        meta = [(file, file.split('/')[3],
                 librosa.get_duration(filename=file), 
                 librosa.get_samplerate(file)
                ) for file in tqdm(file_list)]
    else:
        assert False, f'Unknown mode {mode}'
    return meta

def mel_random_masking(tensor, masking_ratio=0.1, mel_min=-12):

    mask = torch.rand(tensor.shape) > masking_ratio

    masked_tensor = torch.mul(tensor, mask)

    masked_tensor += ~mask * mel_min

    return masked_tensor

def apply_t_shift(tensor, mel_min=-12, T=10):
    
    _, MF = tensor.shape
    
    t = torch.randint(0, T, [1])
    
    shift_tensor = torch.ones(t, MF) * mel_min

    tensor = torch.cat((shift_tensor, tensor), axis=0)
    
    return tensor

def normalize_tensor(tensor, min_v=-12, max_v=0):
    center_v = (max_v - min_v) / 2
    tensor = tensor / center_v  + 1
    return tensor
    
def plot_mel_spectrograms(mel_tensor, keyword=''):

    B, M, T = mel_tensor.shape

    num_x = int(np.sqrt(B))
    num_y = int(B / num_x)

    fig, axes = plt.subplots(num_x, num_y, sharex=True, sharey=True, figsize=(24, 8), dpi=300)
    axes = axes.flatten()

    for i in range(B):
        im = axes[i].imshow(mel_tensor[i, :, :], origin='lower', aspect='auto')

    plt.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(f'mel_sample_{keyword}.png')
    plt.close()

    return

def collate_function(pairs, speaker_table):

    mels = list()
    speakers = list()
    mel_lengths = list()

    B = len(pairs)

    for pair in pairs:
        # (wav_file, clean_script, clean_jamos, tag, len(clean_script), len(clean_jamos), wav_file_dur)
        wav_file = pair[0]
        speaker =  pair[1]
        npy_file = wav_file.replace('.wav', '.npy')
        if not os.path.isfile(npy_file):
            mel = MEL2SAMPWAVEGLOW.get_mel(wav_file).T # (MB, T) -> (T, MB)
            np.save(npy_file, mel)
        else:
            mel = torch.tensor(np.load(npy_file)) # (T, MB)
        mel = apply_t_shift(mel, MEL_MIN)
        mel = mel_random_masking(mel, MASKING_RATIO, MEL_MIN)
        mel = normalize_tensor(mel, MEL_MIN)
        mels.append(mel) 
        mel_lengths.append(mel.shape[0])
        speakers.append(speaker_table[speaker])

    mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-1).transpose(1, 2) # (B, T, MB) -> (B, MB, T)
    mel_lengths = torch.tensor(mel_lengths)
    speakers = torch.tensor(speakers)

    return mel_tensor, mel_lengths, speakers

def write_to_csv(meta_data, file_name):
    with open(f'{file_name}', 'w') as f:
        csv_writer = csv.writer(f)
        for meta in meta_data:
            csv_writer.writerow(meta)
    
    return

def read_from_csv(file_name):
    with open(f'{file_name}', 'r') as f:
        csv_reader = csv.reader(f)
        meta = [(line[0], line[1], float(line[2]), float(line[3])) for line in tqdm(csv_reader)]
             
    return meta

class SpeakerDict():

    def __init__(self, speakers):
        self.speaker_array = sorted(speakers)
        self.speaker_dict = {s: i for i, s in enumerate(self.speaker_array)}

    def __getitem__(self, key):

        if isinstance(key, int):
            return self.speaker_array[key]

        elif isinstance(key, str):
            return self.speaker_dict[key]

        else:
            assert False, f'Invalid key for SpeakerDict {key}'

    def __len__(self):
        a = len(self.speaker_array)
        b = len(self.speaker_dict)
        assert a == b, f'{a} != {b}'
        return a

    def decode_speaker_tensor(self, tensor):
        return [self.speaker_array[v] for v in tensor]

def build_speaker_dict(meta):

    speakers = list(set([m[1] for m in meta]))
    speaker_dict =  SpeakerDict(speakers)

    return speaker_dict

def load_meta(keyword='vox1'):

    if keyword == 'vox1':
        wav_files_dev = sorted(glob('VoxCeleb1/vox1_dev' + '/*/*/*.wav'))
        print(f'Len. wav_files_dev {len(wav_files_dev)}')

        if not os.path.isfile('vox1_dev.csv'):
            dev_meta = struct_meta(wav_files_dev)
            write_to_csv(dev_meta, 'vox1_dev.csv')
        else:
            dev_meta = read_from_csv('vox1_dev.csv')

        wav_files_test = sorted(glob('VoxCeleb1/vox1_test' + '/*/*/*.wav'))
        print(f'Len. wav_files_test {len(wav_files_test)}')

        if not os.path.isfile('vox1_test.csv'):
            test_meta = struct_meta(wav_files_test)
            write_to_csv(test_meta, 'vox1_test.csv')
        else:
            test_meta = read_from_csv('vox1_test.csv')
    elif keyword == 'vox2':
        # TODO
        pass
    else:
        assert False, f'Wrong Keyword {keyword}'

    return dev_meta, test_meta

def get_dataloader(keyword='vox1', t_thres=19):
    dev_meta, test_meta = load_meta(keyword)
    
    test_meta = [meta for meta in tqdm(test_meta) if meta[2] < t_thres]
    dev_meta = [meta for meta in tqdm(dev_meta) if meta[2] < t_thres]
    
    test_speakers = build_speaker_dict(test_meta)
    dev_speakers = build_speaker_dict(dev_meta)

    # def dev_collator(pairs):
    #     return collate_function(pairs, dev_speakers)

    # def test_collator(pairs):
    #     return collate_function(pairs, test_speakers)

    # dataset_dev = DataLoader(dev_meta, batch_size=BATCH_SIZE, 
    #                            shuffle=True, num_workers=NUM_WORKERS,
    #                            collate_fn=dev_collator)

    dataset_dev = DataLoader(dev_meta, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=partial(collate_function, speaker_table=dev_speakers))

    # dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
    #                           shuffle=False, num_workers=NUM_WORKERS,
    #                           collate_fn=lambda x: collate_function(x, test_speakers),
    #                           drop_last=True)
    # AttributeError: Can't pickle local object 'get_dataloader.<locals>.<lambda>'
    # AttributeError: Can't pickle local object 'get_dataloader.<locals>.dev_collator'

    # dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
    #                         shuffle=False, num_workers=NUM_WORKERS,
    #                         collate_fn=test_collator,
    #                         drop_last=True)

    dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=NUM_WORKERS,
                        collate_fn=partial(collate_function, speaker_table=test_speakers),
                        drop_last=True)

    return dataset_dev, dataset_test, dev_speakers, test_speakers

def main():

    wav_files_dev = sorted(glob('VoxCeleb1/vox1_dev' + '/*/*/*.wav'))
    print(f'Len. wav_files_dev {len(wav_files_dev)}')

    if not os.path.isfile('vox1_dev.csv'):
        dev_meta = struct_meta(wav_files_dev)
        write_to_csv(dev_meta, 'vox1_dev.csv')
    else:
        dev_meta = read_from_csv('vox1_dev.csv')

    wav_files_test = sorted(glob('VoxCeleb1/vox1_test' + '/*/*/*.wav'))
    print(f'Len. wav_files_test {len(wav_files_test)}')

    if not os.path.isfile('vox1_test.csv'):
        test_meta = struct_meta(wav_files_test)
        write_to_csv(test_meta, 'vox1_test.csv')
    else:
        test_meta = read_from_csv('vox1_test.csv')

    # print('dev_meta fs: ', set(meta[3] for meta in dev_meta)) # 16000
    # print('test_meta fs: ', set(meta[3] for meta in test_meta)) # 16000

    test_meta = [meta for meta in tqdm(test_meta) if meta[2] < T_THRES]

    dev_meta = [meta for meta in tqdm(dev_meta) if meta[2] < T_THRES]

    test_speakers = build_speaker_dict(test_meta)

    dev_speakers = build_speaker_dict(dev_meta)

    print(f'Test speakers {len(test_speakers)}')
    print(f'Dev speakers {len(dev_speakers)}')

    dataset_dev = DataLoader(dev_meta, batch_size=BATCH_SIZE, 
                               shuffle=True, num_workers=NUM_WORKERS,
                               collate_fn=lambda x: collate_function(x, dev_speakers))

    dataset_test = DataLoader(test_meta, batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=NUM_WORKERS,
                              collate_fn=lambda x: collate_function(x, test_speakers),
                              drop_last=True)

    '''
    https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
    1. collate_fn = lambda b, params=params: my_collator_with_param(b, params)
    2. create a class that contains a __call__ method which will be passed to your DataLoader
    '''

    for mels, mel_length, speakers in tqdm(dataset_dev):
        print(mels.shape)
        print(speakers)

        print(dev_speakers.decode_speaker_tensor(speakers))
        # plot_mel_spectrograms(mels, 'dev')

        # break 
        # pass
        break

    for mels, mel_length, speakers in tqdm(dataset_test):
        print(mels.shape)
        print(speakers)
        # plot_mel_spectrograms(mels, 'test')

        # break
        # pass
        break

    # speakers = dev_speakers | test_speakers

    # print(f'{len(dev_speakers)} + {len(test_speakers)}  = {len(speakers)}')

    # m4a_files_dev = sorted(glob('VoxCeleb2/dev/aac' + '/*/*/*.m4a'))
    # print(f'Len. m4a_files_dev {len(m4a_files_dev)}')

    # m4a_files_test = sorted(glob('VoxCeleb2/test/aac' + '/*/*/*.m4a'))
    # print(f'Len. m4a_files_test {len(m4a_files_test)}')

    # Len. wav_files_dev 148642
    # Len. wav_files_test 4874
    # Len. m4a_files_dev 1092009
    # Len. m4a_files_test 36237




if __name__ == "__main__":
    main()