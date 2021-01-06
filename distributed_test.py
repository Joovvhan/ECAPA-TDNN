import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from prepare_batch_loader import get_dataloader
from tensorboardX import SummaryWriter
import json
import sys
import os
import argparse
from functools import partial

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import defaultdict

global_scope = sys.modules[__name__]

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        # print(f'{key} == {hp[key]}')

from main import ECAPA_TDNN, get_grad_norm, cor_matrix_to_plt_image, save_checkpoint, load_checkpoint, alpha_matrix_to_plt_image

def get_grad_norm_dict(model):


    total_norm = dict()
    for key, value in zip(model.state_dict(), model.parameters()):

        if value.grad is not None:
            param_grad_norm = value.grad.data.norm(2)
            # total_norm[key] = param_grad_norm.item()

            param_norm = value.data.norm(2)
            total_norm[key] = [param_grad_norm.item(), param_norm.item()]
        else:
            total_norm[key] = [-1, -1]

    return total_norm

def process(rank, world_size, run_name=None):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = rank

    dataset_dev, dataset_test, dev_speakers, test_speakers = get_dataloader('vox1', 19)

    model = ECAPA_TDNN(len(dev_speakers), device).to(device)

    # model = ECAPA_TDNN(len(dev_speakers), device)
    model = DDP(model, find_unused_parameters=True, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_func = nn.NLLLoss()

    # input_tensor = torch.rand(B, M, T)
    # ground_truth_tensor = torch.randint(0, T, (B,))

    if rank == 0:
        summary_writer = SummaryWriter(run_name)

    # model.train()

    # torch.autograd.set_detect_anomaly(True)

    loss_list = list()
    acc_list = list()
    # gradient_norm_list = list()
    step = 0

    for epoch in range(NUM_EPOCH):

        ############

        # model.eval()
        # if rank == 0:
        #     embedding_holder = defaultdict(list)
        # for mels, mel_length, speakers in tqdm(dataset_test):
        #     h_tensor, info_tensors = model(mels.to(device), infer=True) # (B, NUM_SPEAKERS)
            
        #     if rank == 0:
        #         for h, s in zip(h_tensor.detach().cpu(), speakers):
        #             embedding_holder[s.item()].append(h)

        # if rank == 0:
        #     for key in embedding_holder:
        #         print(key, len(embedding_holder[key]))

        ###########

        model.train()
        for mels, mel_length, speakers in tqdm(dataset_dev):
            optimizer.zero_grad()
            pred_tensor, info_tensors = model(mels.to(device), speakers.to(device)) # (B, NUM_SPEAKERS)
            loss = loss_func(pred_tensor, speakers.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM) 

            # grad_norm_dict = get_grad_norm_dict(model)

            optimizer.step()

            step += 1

            if rank == 0:

                loss_list.append(loss.item())
                prediction = torch.argmax(pred_tensor, axis=-1)
                acc = (torch.sum((prediction == speakers.to(device)), dtype=torch.float32)/len(speakers)).detach().cpu().numpy()
                acc_list.append(acc)

                # gradient_norm_list.append(get_grad_norm(model))

                if step % CHECKPOINT_STEPS == 0:

                    save_checkpoint(model, optimizer, step, summary_writer.logdir)

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
                    g = model.state_dict()['module.speaker_embedding.weight_g'].detach()
                    v = model.state_dict()['module.speaker_embedding.weight_v'].detach()

                    n = v / g
                    
                    cor_mat = torch.matmul(n, n.T) # (H, W)
                    print(torch.max(cor_mat), torch.min(cor_mat))
                    matrix_image = cor_matrix_to_plt_image(cor_mat.cpu(), step)
                    summary_writer.add_image('train/speaker_correlation', matrix_image, step)

                    alpha_tensor = info_tensors[0]
                    matrix_image = alpha_matrix_to_plt_image(alpha_tensor, step)
                    summary_writer.add_image('train/alpha_matrix', matrix_image, step)

        model.eval()

        if rank == 0: embedding_holder = defaultdict(list)
        for mels, mel_length, speakers in tqdm(dataset_test):
            h_tensor, info_tensors = model(mels.to(device), infer=True) # (B, NUM_SPEAKERS)
            
            if rank == 0:
                for h, s in zip(h_tensor.detach().cpu(), speakers):
                    embedding_holder[s.item()].append(h)

        if rank == 0:
            for key in embedding_holder:
                print(key, len(embedding_holder[key]))

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training options')
    parser.add_argument('--run_name', metavar='N', type=str, default=None)
    args = parser.parse_args()

    print(args)

    if args.run_name is not None :
        args.run_name = os.path.join('runs', args.run_name)
    else: 
        args.run_name = None

    world_size = 2

    # world_size = 1

    # torch.autograd.set_detect_anomaly(True)

    mp.spawn(partial(process, run_name=args.run_name),
        args=(world_size,),
        nprocs=world_size,
        join=True)

    dist.destroy_process_group()