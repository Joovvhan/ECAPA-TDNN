import torch
from torch import optim
from prepare_batch_loader import wav2mel_tensor
import json
import sys
import os
import argparse
from glob import glob
import numpy as np

global_scope = sys.modules[__name__]

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        # print(f'{key} == {hp[key]}')

from main import ECAPA_TDNN, load_checkpoint

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training options')
    parser.add_argument('--run_name', metavar='N', type=str, default='1e-5-all-speakers-HR=16-re')
    args = parser.parse_args()

    print(args)

    if args.run_name is not None :
        args.run_name = os.path.join('runs', args.run_name)
    else: 
        assert False, 'Empty run name'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = ECAPA_TDNN(len(dev_speakers), device).to(device)
    model = ECAPA_TDNN(1211, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model, optimizer, step = load_checkpoint(model, optimizer, args.run_name, rank='cpu')

    model.eval()
    
    wav_files = sorted(glob('./wavs/positive/*.wav'))
    mels = wav2mel_tensor(wav_files)
    h_tensor, info_tensors = model(mels.to(device), infer=True) # (B, NUM_SPEAKERS)
    np.save('positive_embedding.npy', h_tensor.detach().numpy())

    wav_files = sorted(glob('./wavs/negative/*.wav'))
    mels = wav2mel_tensor(wav_files)
    h_tensor, info_tensors = model(mels.to(device), infer=True) # (B, NUM_SPEAKERS)
    np.save('negative_embedding.npy', h_tensor.detach().numpy())