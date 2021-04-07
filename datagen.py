import os
import random
import shutil
import time
import argparse

def datasplit(args) :
    print('run datasplit~~')
    start_time=time.time()
    # folder0to99=os.listdir('/mnt/nas2/hm/hm_fusionFR/data/data1_clean1')
    folder0to99=os.listdir('/mnt/nas2/kkm/face_recognition/train_point')


    os.mkdir('/mnt/nas2/hm/hm_fusionFR/data/'+args.save_foldername+'/train')
    os.mkdir('/mnt/nas2/hm/hm_fusionFR/data/'+args.save_foldername+'/test')
    for i in folder0to99 :
        os.mkdir('/mnt/nas2/hm/hm_fusionFR/data/'+args.save_foldername+'/train/'+i)
        os.mkdir('/mnt/nas2/hm/hm_fusionFR/data/'+args.save_foldername+'/test/'+i)

    train_file_dir = '/mnt/nas2/hm/hm_fusionFR/data/data1/'
    files = os.listdir(train_file_dir)

    print('zz1')
    for i in files:
        file_lst = os.listdir(train_file_dir + i)
        if len(file_lst) == 0:
            continue

        file_lst = [i.split('.')[0] for i in file_lst]
        # ll = len(file_lst)
        file_lst = list(set(file_lst))

        n = len(file_lst) if len(file_lst) < args.train_nums else args.train_nums

        tr = random.sample(file_lst, n)
        te = [x for x in file_lst if x not in tr]
        tr = [i + '.jpg' for i in tr] + [i + '.ply' for i in tr]
        te = [i + '.jpg' for i in te] + [i + '.ply' for i in te]

        for j in tr:
            from_ = train_file_dir + i + '/' + j
            to_ = '/mnt/nas2/hm/hm_fusionFR/data/'+args.save_foldername+'/train/' + i
            shutil.copy(from_, to_)

        for k in te:
            from_ = train_file_dir + i + '/' + k
            to_ = '/mnt/nas2/hm/hm_fusionFR/data/' + args.save_foldername + '/test/' + i
            shutil.copy(from_, to_)

    print("datasplit complete !!   time : ",time.time()-start_time," seconds ")

if __name__ == '__main__':
    print('run datasplit')
    # datasplit(train_nums=5)

    parser = argparse.ArgumentParser(description = 'Argparse')
    parser.add_argument(
      '--train_nums', '-trn',
      type=int,
        default=3,
        help='train_nums'
    )
    parser.add_argument(
      '--save_foldername', '-name',
      type=str,
        default="data_num3",
        help='save folder name'
    )

    args = parser.parse_args()
    datasplit(args)
