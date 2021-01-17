import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from prepare_batch_loader import get_dataloader

from tensorboardX import SummaryWriter

import json

import sys

import matplotlib.pyplot as plt

import os

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

# HYPER_RADIUS = 1.0
HYPER_RADIUS = 16.0

AMM_MARGIN = 0.2

LOGGING_STEPS = 300

NUM_EPOCH = 24

global_scope = sys.modules[__name__]

CONFIGURATION_FILE='config.json'

ONCE = True

with open(CONFIGURATION_FILE) as f:
    data = f.read()
    json_info = json.loads(data)

    hp = json_info["hp"]

    for key in hp:
        setattr(global_scope, key, hp[key])
    #    print(f'{key} == {hp[key]}')

    # model_parameters = json_info["mp"]

def label2mask(label, h):
    B = len(label)
    H = h
    mask = torch.zeros([B, H], requires_grad=False)
    mask[torch.arange(B), label] = 1
    # for i, l in enumerate(label):
    #     mask[i, l] = 1.0
        
    return mask

def cor_matrix_to_plt_image(matrix_tensor, step, apply_diagonal_zero=True):
    
    if apply_diagonal_zero:
        for i in range(len(matrix_tensor)):
            matrix_tensor[i, i] = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    axes[0].set_title(f'Speaker Embedding Correlation #{step:07d}', fontsize=24)
    im = axes[0].imshow(matrix_tensor)
    fig.colorbar(im, ax=axes[0])

    axes[1].set_title(f'Normalized Correlation #{step:07d}', fontsize=24)
    im = axes[1].imshow(matrix_tensor)
    im.set_clim([-1, 1])
    fig.colorbar(im, ax=axes[1])

    axes[2].hist(matrix_tensor.numpy().flatten(), 
             bins=np.arange(-1, 1, 0.05), 
             alpha = 0.5, density=True)
    '''
    axes[2].hist(matrix_tensor.flatten(), 
             bins=np.arange(-1, 1, 0.05), 
             alpha = 0.5, density=True)

    /opt/conda/lib/python3.8/site-packages/numpy/lib/histograms.py:905: 
    RuntimeWarning: invalid value encountered in true_divide
    return n/db/n.sum(), bin_edges
    '''
    axes[2].set_title(f'Correlation Distribution #{step:07d}', fontsize=24)

    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    plt.close()

    return image_array

def alpha_matrix_to_plt_image(alpha_matrix, input_mel, step):
    '''
    input_mel (B, MB, T)
    '''
    
    fig, axes = plt.subplots(3, 1, figsize=(24, 12))
    alpha_matrix_slice = alpha_matrix[0, :, :].T
    # alpha_mean = np.mean(alpha_matrix_slice, axis=0)
    alpha_mean = torch.mean(alpha_matrix_slice, axis=0)
    
    im = axes[0].imshow(alpha_matrix_slice, aspect='auto')
    axes[0].set_title(f'Alpha Matrix #{step:07d}')
    fig.colorbar(im, ax=axes)

    axes[1].plot(alpha_mean)
    axes[1].set_xlim([0, len(alpha_mean)])
    axes[1].set_title(f'Channel Mean Alpha')

    axes[2].imshow(input_mel[0, :, :], origin='lower', aspect='auto')

    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    plt.close()

    return image_array

def inference_embeddings_to_plt_hist(embedding_holder, step):
    fig = plt.figure(figsize=(6, 6))
    plt.title(f'Inference Similarity #{step:07d}', fontsize=18)

    def mean_method(input_vector_list):
        mean_vector = np.mean(input_vector_list, axis=0)
        return mean_vector/np.linalg.norm(mean_vector)

    mean_embedding = {key: mean_method(np.stack(embedding_holder[key])) for key in embedding_holder}

    similarity_score_list = list()
    dissimilarity_score_list = list()

    for key1 in mean_embedding:
        for key2 in mean_embedding:
            mean_vector = mean_embedding[key1]
            embeddings = embedding_holder[key2]
            scores = np.matmul(embeddings, mean_vector)
            
            if key1 == key2:
                similarity_score_list.extend(scores)
            else:
                dissimilarity_score_list.extend(scores)

    plt.hist(similarity_score_list, 
            bins=np.arange(-1, 1, 0.05), 
            alpha = 0.5, label='sim', density=True)

    plt.hist(dissimilarity_score_list, 
            bins=np.arange(-1, 1, 0.05), 
            alpha = 0.5, label='dis', density=True)
    plt.legend()

    # plt.clim([0, 1])
    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    plt.close()

    return image_array

class AttentiveStatPooling(nn.Module):

    def __init__(self):
        super(AttentiveStatPooling, self).__init__()
        self.linear1 = nn.Linear(3 * H_CONV, H_ATT)
        self.linear2 = nn.Linear(H_ATT, 3 * H_CONV)   

    def forward(self, input_tensor, mask_tensor=None):
        h = input_tensor.transpose(1, 2) # (B, H, L) =>  (B, L, H) 
        tensor = F.relu(self.linear1(h)) # (B, L, 128) 
        e_tensor = self.linear2(tensor) # (B, L, 1536) 
        # a_tensor = F.softmax(e_tensor, dim=-1) # (B, L, 1536)
        a_tensor = F.softmax(e_tensor, dim=1) # (B, L, 1536)

        a_h_tensor = torch.mul(a_tensor, h) # (B, L, 1536)

        h_mean = torch.sum(a_h_tensor, dim=1, keepdim=False) # (B, H)
        h_mean_square = torch.mul(h_mean, h_mean) # (B, H)

        h_square = torch.mul(h, h) # (B, L, H)
        weighted_h_mean_square = torch.mul(a_tensor, h_square) # (B, L, H)

        weighted_square = torch.sum(weighted_h_mean_square, dim=1, keepdim=False) # (B, H)

        # neg_tensor = weighted_square - h_mean_square
        neg_tensor = (weighted_square - h_mean_square).clamp(min=1e-4)  
        
        if (neg_tensor < 0).any(): print("########## Negative value in Negative Tensor")
        sigma = torch.sqrt(neg_tensor) # (B, H) - (B, H)
        if torch.isnan(sigma).any(): 
            print("########## NaN in Sigma Tensor")
            print(torch.min(neg_tensor))

        tensor = torch.cat((h_mean, sigma), axis=1)

        return tensor, a_tensor

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
                tensor = tensor + last_tensor
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

        # tensor = tensor + se # Gradient Explodes!!!
        
        # tensor = se + input_tensor 

        tensor = se

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
        
        self.m = torch.tensor(AMM_MARGIN, requires_grad=False).to(device)

        self.num_speakers = NUM_SPEAKERS

        self.device = device

    def forward(self, input_tensor, ground_truth_tensor=None, mask_tensor=None, infer=False):

        if infer:
            return self.infer(input_tensor, mask_tensor=None)

        if torch.isnan(input_tensor).any(): print("NaN in input_tensor")
        tensor = self.conv1d_in(input_tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)
        if torch.isnan(tensor).any(): print("NaN in first batch norm")

        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_1 = tensor_1 + tensor
        if torch.isnan(tensor_1).any(): print("NaN in tensor_1")
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_2 = tensor_2 + tensor_1 + tensor
        if torch.isnan(tensor_2).any(): print("NaN in tensor_2")
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)
        tensor_3 = tensor_3 + tensor_2 + tensor_1 + tensor
        if torch.isnan(tensor_3).any(): print("NaN in tensor_3")

        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        # tensor = torch.cat([tensor_1, tensor_1, tensor_1], axis=1) # (B, 3C, T)
        tensor = self.conv1d_out(tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)

        '''
        The dimension of the bottleneck in the SE-Block and the attention module is set to 128.
        '''

        tensor, a_tensor = self.attentive_stat_pooling(tensor, mask_tensor) # (B, H, L) =>  (B, H) 
        if torch.isnan(tensor).any(): print("NaN in attentive_stat_pooling")

        tensor = self.batchnorm_2(tensor)

        tensor = self.fc(tensor)

        tensor = self.batchnorm_3(tensor) # (B, H)

        tensor_g = torch.norm(tensor, dim=1, keepdim=True)
        normalized_tensor = tensor / tensor_g
        if torch.isnan(normalized_tensor).any(): print("NaN in normalization")

        tensor = self.speaker_embedding(normalized_tensor)
        layer_g = self.speaker_embedding.weight_g
        tensor = tensor / layer_g.squeeze(1)

        if torch.isnan(tensor).any(): print("NaN in normalizing speaker_embedding")

        # v * torch.cos(self.m) + (1 - v * v) * torch.sin(self.m)
        # (masked v)* torch.cos(self.m)
        # + torch.sqrt(masked - (masked v) * (masked v)) * torch.sin(self.m)
        # - (masked v)

        # Additive Margin
        mask = label2mask(ground_truth_tensor, self.num_speakers).to(self.device)

        masked_embedding = tensor * mask
        modified_angle = torch.acos(masked_embedding) + self.m * mask
        modified_angle = torch.clamp(modified_angle, 0, torch.acos(torch.tensor(-1.0)))
        modified_cos = torch.cos(modified_angle)
        tensor = tensor + modified_cos - masked_embedding

        if torch.isnan(tensor).any(): print("NaN in maringalizing")

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

        info_tensors = (a_tensor.detach().cpu(), )

        return prediction, info_tensors

    def infer(self, input_tensor, mask_tensor=None):
        tensor = self.conv1d_in(input_tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)
        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_1 = tensor_1 + tensor
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_2 = tensor_2 + tensor_1 + tensor
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)
        tensor_3 = tensor_3 + tensor_2 + tensor_1 + tensor
        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        tensor = self.conv1d_out(tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor, a_tensor = self.attentive_stat_pooling(tensor, mask_tensor) # (B, H, L) =>  (B, H) 
        tensor = self.batchnorm_2(tensor)
        tensor = self.fc(tensor)
        tensor = self.batchnorm_3(tensor) # (B, H)
        tensor_g = torch.norm(tensor, dim=1, keepdim=True)
        normalized_tensor = tensor / tensor_g
        # tensor = self.speaker_embedding(normalized_tensor)
        # layer_g = self.speaker_embedding.weight_g
        # tensor = tensor / layer_g.squeeze(1)

        # mask = label2mask(ground_truth_tensor, self.num_speakers).to(self.device)

        # masked_embedding = tensor * mask
        # modified_angle = torch.acos(masked_embedding) + self.m * mask
        # modified_angle = torch.clamp(modified_angle, 0, torch.acos(torch.tensor(-1.0)))
        # modified_cos = torch.cos(modified_angle)
        # tensor = tensor + modified_cos - masked_embedding

        # tensor = self.scale * tensor

        # prediction = F.log_softmax(tensor, dim=1) # (B, NUM_SPEAKERS)

        info_tensors = (a_tensor.detach().cpu(), )

        return normalized_tensor, info_tensors

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

def save_checkpoint(model, optimizer, step, path, checkpoint_name='checkpoint.pt'):

    checkpoint_path = os.path.join(path, checkpoint_name)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    return

def load_checkpoint(model, optimizer, path, rank=None, checkpoint_name='checkpoint.pt'):

    checkpoint_path = os.path.join(path, checkpoint_name)

    if not os.path.isfile(checkpoint_path):
        return model, optimizer, 0

    if rank is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)


    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']

    return model, optimizer, step

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
                    if i == 0:
                        g = p.detach()
                    elif i == 1:
                        v = p.detach()

                n = v / g
                
                cor_mat = torch.matmul(n, n.T) # (H, W)
                # cor_mat = torch.matmul(s, s.T).unsqueeze(0) # (1, H, W)
                print(torch.max(cor_mat), torch.min(cor_mat))
                matrix_image = cor_matrix_to_plt_image(cor_mat.cpu(), step)
                summary_writer.add_image('train/speaker_correlation', matrix_image, step)
                # summary_writer.add_image('train/speaker_correlation', cor_mat, step)


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
