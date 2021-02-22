import os
import shutil

def copy_file_calib():

    source_seq = '/media/ddd/data3/rs-128/calib.txt'
    desti = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/data/rs128/training/calib/'
    for idx in range(0,251):
        desti_f = os.path.join(desti,str(idx).zfill(6)+'.txt')
        shutil.copyfile(source_seq, desti_f)
        print('seq:',idx,'over')


copy_file_calib()