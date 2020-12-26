import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from prepare_batch_loader import get_dataloader
from tensorboardX import SummaryWriter
import json
import sys
import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

global_scope = sys.modules[__name__]

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        # print(f'{key} == {hp[key]}')

from main import ECAPA_TDNN, get_grad_norm, cor_matrix_to_plt_image

def process(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = rank

    dataset_dev, dataset_test, dev_speakers, test_speakers = get_dataloader('vox1', 19)

    model = ECAPA_TDNN(len(dev_speakers), device).to(device)
    # model = ECAPA_TDNN(len(dev_speakers), device)
    model = DDP(model, find_unused_parameters=True, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_func = nn.NLLLoss()

    # input_tensor = torch.rand(B, M, T)
    # ground_truth_tensor = torch.randint(0, T, (B,))

    if rank == 0:
        summary_writer = SummaryWriter()

    model.train()

    # torch.autograd.set_detect_anomaly(True)

    loss_list = list()
    acc_list = list()
    # gradient_norm_list = list()
    step = 0

    for epoch in range(NUM_EPOCH):

        for mels, mel_length, speakers in tqdm(dataset_dev):

            optimizer.zero_grad()
            pred_tensor = model(mels.to(device), speakers.to(device)) # (B, NUM_SPEAKERS)
            loss = loss_func(pred_tensor, speakers.to(device))
            loss.backward()
            optimizer.step()

            step += 1

            if rank == 0:

                loss_list.append(loss.item())
                prediction = torch.argmax(pred_tensor, axis=-1)
                acc = (torch.sum((prediction == speakers.to(device)), dtype=torch.float32)/len(speakers)).detach().cpu().numpy()
                acc_list.append(acc)

                # gradient_norm_list.append(get_grad_norm(model))

                if step % LOGGING_STEPS == 0:
                    # print(loss_list)
                    loss_mean = np.mean(loss_list)
                    # loss_mean = np.nanmean(loss_list)
                    summary_writer.add_scalar('train/loss', loss_mean, step)
                    loss_list = list()

                    acc_mean = np.mean(acc_list)
                    summary_writer.add_scalar('train/acc', acc_mean, step)
                    acc_list = list()

                    # grad_norm_mean = np.mean(gradient_norm_list)
                    # summary_writer.add_scalar('train/grad_norm', grad_norm_mean, step)
                    # gradient_norm_list  = list()

                    summary_writer.add_scalar('train/grad_norm', get_grad_norm(model), step)
                    
                    # for i, p in enumerate(model.speaker_embedding.parameters()):
                    #     if i == 0:
                    #         g = p.detach()
                    #     elif i == 1:
                    #         v = p.detach()

                    # n = v / g
                    
                    # cor_mat = torch.matmul(n, n.T) # (H, W)
                    # print(torch.max(cor_mat), torch.min(cor_mat))
                    # matrix_image = cor_matrix_to_plt_image(cor_mat.cpu(), step)
                    # summary_writer.add_image('train/speaker_correlation', matrix_image, step)


    model.eval()

    return

if __name__ == "__main__":
    
    world_size = 2
    mp.spawn(process,
        args=(world_size,),
        nprocs=world_size,
        join=True)

    dist.destroy_process_group()
