import torch
import math
import time
def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    start = time.time()
    C_25_7 = 6103515625  # 25**7

    L1, a1, b1 = Lab_1[:, :, :, 0], Lab_1[:, :, :, 1], Lab_1[:, :, :, 2]  # b,N,K,1
    L2, a2, b2 = Lab_2[:, :, :, 0], Lab_2[:, :, :, 1], Lab_2[:, :, :, 2]  # b,N,K,1

    C1 = torch.sqrt(a1 ** 2 + b1 ** 2) # b,N,K,1
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2) # b,N,K,1
    C_ave = (C1 + C2) / 2 # b,N,K,1
    G = 0.5 * (1 - torch.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7)))  # b,N,K,1

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = torch.sqrt(a1_ ** 2 + b1_ ** 2)
    C2_ = torch.sqrt(a2_ ** 2 + b2_ ** 2)

    idx = torch.where((b1_==0) & (a1_==0))
    #h1_ = torch.zeros((b1_.shape[0],b1_.shape[1],b1_.shape[2])).cuda()
    h1_ = torch.atan2(b1_, a1_)
    h1_[idx] = 0
    '''
    idx = torch.where((a1_ == 0) & (b1_ != 0))
    h1_[idx] = torch.atan2(b1_[idx], a1_[idx])
    idx = torch.where((a1_ > 0))
    #h1_[idx] = torch.atan2(b1_[idx], a1_[idx])
    '''
    idx = torch.where((a1_ < 0))
    h1_[idx] = h1_[idx] + 2* math.pi

    idx = torch.where((b2_ == 0) & (a2_ == 0))
    h2_ = torch.atan2(b2_, a2_)
    h2_[idx] = 0
    #idx = torch.where((a2_ == 0) & (b2_ != 0))
    #h2_[idx] = torch.atan2(b2_[idx], a2_[idx])

    idx = torch.where((a2_ < 0))
    h2_[idx] = h2_[idx] + 2 * math.pi
    #idx = torch.where((a2_ > 0))
    #h2_[idx] = torch.atan2(b2_[idx], a2_[idx])


    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_

    idx = torch.where((C1_ == 0) & (C2_ == 0))
    dh_[idx] = 0
    idx = torch.where((dh_ > math.pi))
    dh_[idx] = dh_[idx] - 2* math.pi
    idx = torch.where((dh_ < math.pi))
    dh_[idx] = dh_[idx] + 2 * math.pi

    dH_ = 2 * torch.sqrt(C1_ * C2_) * torch.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    h_ave = (h1_ + h2_) / 2
    #idx = torch.where((C1C2 != 0) & (_dh <=math.pi))
    #h_ave[idx] = (h1_[idx] + h2_[idx]) / 2
    idx = torch.where((C1C2 != 0) & (_dh > math.pi) &  (_sh < 2 * math.pi))
    h_ave[idx] = h_ave[idx] + math.pi
    idx = torch.where((C1C2 != 0) & (_dh > math.pi) & (_sh < 2 * math.pi))
    h_ave[idx] = h_ave[idx] + math.pi
    idx = torch.where((C1C2 != 0) & (_dh > math.pi) & (_sh >= 2 * math.pi))
    h_ave[idx] = h_ave[idx] - math.pi
    idx = torch.where((C1C2 == 0))
    h_ave[idx] = h_ave[idx]*2


    T = 1 - 0.17 * torch.cos(h_ave - math.pi / 6) + 0.24 * torch.cos(2 * h_ave) + 0.32 * torch.cos(
        3 * h_ave + math.pi / 30) - 0.2 * torch.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    idx = torch.where((h_ave_deg < 0))
    h_ave_deg[idx] = h_ave_deg[idx] + 360
    idx = torch.where((h_ave_deg >360))
    h_ave_deg[idx] = h_ave_deg[idx] - 360

    dTheta = 30 * torch.exp(-(((h_ave_deg - 275) / 25) ** 2))

    R_C = 2 * torch.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / torch.sqrt(20 + Lm50s)
    R_T = -torch.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = torch.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)
    end = time.time()
    #print('CIEDE2000 :',end-start)
    return dE_00

def CIEDE1994(Lab_1, Lab_2):
    #start = time.time()
    L1, a1, b1 = Lab_1[:, :, :, 0], Lab_1[:, :, :, 1], Lab_1[:, :, :, 2]  # b,N,K,1
    L2, a2, b2 = Lab_2[:, :, :, 0], Lab_2[:, :, :, 1], Lab_2[:, :, :, 2]  # b,N,K,1
    L12 = L1 - L2
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2)  # b,N,K,1
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2)  # b,N,K,1
    C12 = C1 - C2

    a12 = a1 - a2
    b12 = b1 - b2
    H12 = torch.sqrt(a12 ** 2 + b12 ** 2 - C12 **2)

    Sl = 1
    Kl = 1
    Kc = 1
    Kh = 1
    K1 = 0.045
    K2 = 0.015

    Sc = 1 + K1*C1
    Sh = 1 + K2*C2

    El = L12/(Kl*Sl)
    Ec = C12/(Kc*Sc)
    Eh = H12/(Kh*Sh)

    E = torch.sqrt(El**2 + Ec**2 + Eh**2)
    # end = time.time()
    #print('CIEDE2000 :', end - start)
    return  E
