import torch

a = torch.randn([3,100,3])
index = (torch.LongTensor([0,1]),torch.LongTensor([1,2])
a.index_put_(index), torch.Tensor([1,1]))