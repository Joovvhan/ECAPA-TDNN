import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from prepare_batch_loader import get_dataloader

from tensorboardX import SummaryWriter

import json

import sys

B, M, T = 4, 80, 17

'''
We study two setups of the proposed ECAPA-TDNN architecture with either 
512 or 1024 channels in the convolutional frame layers. 
The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
The scale dimension s in the Res2Block [16] is set to 8. 
The number of nodes in the final fully-connected layer is 192. 
The performance of this system will be compared to the baselines described in Section 2.
'''

H_CONV = 512

H_ATT = 128

H_FC = 192

# NUM_SPEAKERS = 6000

HYPER_RADIUS = 1.0

AMM_MARGIN = 0.2

LOGGING_STEPS = 300

NUM_EPOCH = 24

global_scope = sys.modules[__name__]

CONFIGURATION_FILE='config.json'

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
        print(f'{key} == {hp[key]}')

    # model_parameters = json_info["mp"]

def label2mask(label, h):
    B = len(label)
    H = h
    mask = torch.zeros([B, H], requires_grad=False)
    mask[torch.arange(B), label] = 1
    # for i, l in enumerate(label):
    #     mask[i, l] = 1.0
        
    return mask

class AttentiveStatPooling(nn.Module):

    def __init__(self):
        super(AttentiveStatPooling, self).__init__()
        self.linear1 = nn.Linear(3 * H_CONV, H_ATT)
        self.linear2 = nn.Linear(H_ATT, 3 * H_CONV)   

    def forward(self, input_tensor):
        h = input_tensor.transpose(1, 2) # (B, H, L) =>  (B, L, H) 
        tensor = F.relu(self.linear1(h)) # (B, L, 128) 
        e_tensor = self.linear2(tensor) # (B, L, 1536) 
        a_tensor = F.softmax(e_tensor, dim=-1) # (B, L, 1536)

        h_mean = torch.sum(torch.mul(h, a_tensor), dim=1, keepdim=False) # (B, H)

        h_square = torch.mul(h, h) # (B, L, H)

        weighted_square = torch.sum(torch.mul(a_tensor, h_square), dim=1)

        sigma = torch.sqrt(weighted_square - torch.mul(h_mean, h_mean)) # (B, H) - (B, H)

        tensor = torch.cat((h_mean, sigma), axis=1)

        return tensor

class SE_RES2Block(nn.Module):

    def __init__(self, scale=8, k=3, d=2):
        super(SE_RES2Block, self).__init__()

        self.scale = scale
        self.k = k
        self.d = d
        self.padding = int(self.d * (self.k - 1) / 2)
        self.res2net_hiddens = [int(H_CONV / (2 ** i)) for i in range(self.scale)]
        self.conv1d_expand = nn.Conv1d(H_CONV, H_CONV * self.scale, 1)
        self.batchnorm_expand = nn.BatchNorm1d(H_CONV * self.scale)

        self.res2conv1d_list = nn.ModuleList([nn.Conv1d(H_CONV, H_CONV, self.k, dilation=self.d, padding=self.padding) for i in range(self.scale-1)])
        self.res2batch_norm_list = nn.ModuleList([nn.BatchNorm1d(H_CONV) for i in range(self.scale-1)])

        self.conv1d_collapse = nn.Conv1d(H_CONV * self.scale, H_CONV, 1)
        self.batchnorm_collapse = nn.BatchNorm1d(H_CONV)

        self.fc_1 = nn.Linear(H_CONV, 128)
        self.fc_2 = nn.Linear(128, H_CONV)

    def forward(self, input_tensor):

        tensor = self.conv1d_expand(input_tensor)  # (B, C, T) -> (B, 8C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_expand(tensor)

        tensors = torch.split(tensor, H_CONV, dim=1) # (B, 8C, T) -> (B, C, T)

        tensor_list = []

        assert len(tensors) == len(self.res2conv1d_list) + 1, f'{len(tensors)} != {len(self.res2conv1d_list)} + 1'

        for i, tensor in enumerate(tensors):
            if i > 1:
                tensor += last_tensor
            if i > 0:
                tensor = self.res2conv1d_list[i-1](tensor)
                tensor = F.relu(tensor)
                tensor = self.res2batch_norm_list[i-1](tensor)
            tensor_list.append(tensor)
            last_tensor = tensor 

        tensor = torch.cat(tensor_list, axis=1)

        tensor = self.conv1d_collapse(tensor)  # (B, 8C, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_collapse(tensor)

        '''
        The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
        '''

        z = torch.mean(tensor, dim=2)

        s = torch.sigmoid(self.fc_2(F.relu(self.fc_1(z)))) # (B, C)

        s = torch.unsqueeze(s, dim=2) # (B, C, 1)

        # s = torch.unsqueeze(s, dim=-1)

        se = s * tensor # (B, C, 1) * (B, C, T) = (B, C, T)

        # print(z.shape)
        # print(s.shape)

        tensor = tensor + se

        # tensor += se 
        # RuntimeError: one of the variables needed for gradient computation 
        # has been modified by an inplace operation

        return tensor

class ECAPA_TDNN(nn.Module):

    def __init__(self, NUM_SPEAKERS, device):
        super(ECAPA_TDNN, self).__init__()

        self.conv1d_in = nn.Conv1d(80, H_CONV, 5, padding=2)
        self.batchnorm_in = nn.BatchNorm1d(H_CONV)

        self.se_res2block_1 = SE_RES2Block(8, 3, 2)
        self.se_res2block_2 = SE_RES2Block(8, 3, 3)
        self.se_res2block_3 = SE_RES2Block(8, 3, 4)

        self.conv1d_out = nn.Conv1d(3 * H_CONV, 3 * H_CONV, 1)

        self.attentive_stat_pooling = AttentiveStatPooling()

        self.batchnorm_2 = nn.BatchNorm1d(H_CONV * 3 * 2)

        self.fc = nn.Linear(H_CONV * 3 * 2, H_FC)

        self.batchnorm_3 = nn.BatchNorm1d(H_FC)

        self.speaker_embedding = nn.utils.weight_norm(nn.Linear(H_FC, NUM_SPEAKERS, bias=False), dim=0)

        self.scale = HYPER_RADIUS
        
        self.m = torch.tensor(AMM_MARGIN, requires_grad=False)

        self.num_speakers = NUM_SPEAKERS

        self.device = device

    def forward(self, input_tensor, ground_truth_tensor):

        tensor = self.conv1d_in(input_tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)

        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)

        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        tensor = F.relu(tensor)

        '''
        The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
        '''

        tensor = self.attentive_stat_pooling(tensor) # (B, H, L) =>  (B, H) 

        tensor = self.batchnorm_2(tensor)

        tensor = self.fc(tensor)

        # tensor = self.batchnorm_3(tensor) # (B, H)

        tensor_g = torch.norm(tensor, dim=1, keepdim=True)

        normalized_tensor = tensor / tensor_g

        tensor = self.speaker_embedding(normalized_tensor)

        layer_g = self.speaker_embedding.weight_g

        tensor = tensor / layer_g.squeeze(1)

        # v * torch.cos(self.m) + (1 - v * v) * torch.sin(self.m)
        # (masked v)* torch.cos(self.m)
        # + torch.sqrt(masked - (masked v) * (masked v)) * torch.sin(self.m)
        # - (masked v)

        mask = label2mask(ground_truth_tensor, self.num_speakers).to(self.device)

        masked_embedding = tensor * mask
        modified_angle = torch.acos(masked_embedding) + self.m * mask
        modified_angle = torch.clamp(modified_angle, 0, torch.acos(torch.tensor(-1.0)))
        modified_cos = torch.cos(modified_angle)
        tensor = tensor + modified_cos - masked_embedding

        '''
        https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss
        '''

        # cos_embed = torch.cos(self.m) * masked_embedding
        # masked_embedding_square = masked_embedding * masked_embedding
        # sin_embed_square = mask - masked_embedding_square
        # sin_embed = torch.sin(self.m) * torch.sqrt(sin_embed_square)
        # modified_embedding = cos_embed - sin_embed - masked_embedding

        # modified_embedding = torch.cos(self.m) * masked_embedding - \
        #     torch.sin(self.m) * torch.sqrt(mask - masked_embedding * masked_embedding) - \
        #     masked_embedding


        # modified_embedding = torch.cos(self.m) * masked_embedding - \
        #     masked_embedding
        
        # tensor = tensor + modified_embedding

        tensor = self.scale * tensor

        # for i, label in enumerate(ground_truth_tensor):
        #     t = tensor[i, label].clone()
        #     tensor[i, label] = - t * t + t  + 1 + torch.cos(self.m) + torch.sin(self.m)

        '''
        All systems are trained using AAM-softmax [6, 25] with a margin of 0.2 and 
        softmax prescaling of 30 for 4 cycles.
        '''

        prediction = F.log_softmax(tensor, dim=1) # (B, NUM_SPEAKERS)

        return prediction


# https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
def get_grad_norm(model):

    # This function increases training time twice longer

    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)

    return grad_norm

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_dev, dataset_test, dev_speakers, test_speakers = get_dataloader('vox1', 19)

    model = ECAPA_TDNN(len(dev_speakers), device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_func = nn.NLLLoss()

    # input_tensor = torch.rand(B, M, T)
    # ground_truth_tensor = torch.randint(0, T, (B,))

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
                
                for i, p in enumerate(model.speaker_embedding.parameters()):
                    if i == 1:
                        s = p.detach()
                
                cor_mat = torch.matmul(s, s.T).unsqueeze(0) # (1, H, W)
                print(torch.max(cor_mat), torch.min(cor_mat))
                summary_writer.add_image('train/speaker_correlation', cor_mat, step)


    # for mels, mel_length, speakers in tqdm(dataset_test):
    #     print(mels.shape)
    #     print(speakers)
    #     # plot_mel_spectrograms(mels, 'test')

    #     # break
    #     # pass
    #     break

    model.eval()

    # print(tensor.shape)
    # print(input_tensor.shape)

    # print("main")
    return

if __name__ == "__main__":
    main()
