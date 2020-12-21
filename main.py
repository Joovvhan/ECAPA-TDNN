import torch
from torch import nn
import torch.nn.functional as F

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

        se = s * tensor # (B, C, 1) * (B, C, T) = (B, C, T)

        # print(z.shape)
        # print(s.shape)

        tensor += se

        return tensor

class ECAPA_TDNN(nn.Module):

    def __init__(self):
        super(ECAPA_TDNN, self).__init__()

        self.conv1d_in = nn.Conv1d(80, H_CONV, 5, padding=2)
        self.batchnorm_in = nn.BatchNorm1d(H_CONV)

        self.se_res2block_1 = SE_RES2Block(8, 3, 2)
        self.se_res2block_2 = SE_RES2Block(8, 3, 3)
        self.se_res2block_3 = SE_RES2Block(8, 3, 4)

        self.conv1d_out = nn.Conv1d(3 * H_CONV, 3 * H_CONV, 1)
  


    def forward(self, input_tensor):

        tensor = self.conv1d_in(input_tensor) # (B, M, T) -> (B, C, T)
        tensor = F.relu(tensor)
        tensor = self.batchnorm_in(tensor)

        tensor_1 = self.se_res2block_1(tensor)   # (B, C, T)
        tensor_2 = self.se_res2block_2(tensor_1) # (B, C, T)
        tensor_3 = self.se_res2block_3(tensor_2) # (B, C, T)

        tensor = torch.cat([tensor_1, tensor_2, tensor_3], axis=1) # (B, 3C, T)
        tensor = F.relu(tensor)

        return tensor

def main():

    model = ECAPA_TDNN()

    input_tensor = torch.rand(B, M, T)

    tensor =  model(input_tensor)

    print(tensor.shape)
    # print(input_tensor.shape)

    # print("main")
    return

if __name__ == "__main__":
    main()
