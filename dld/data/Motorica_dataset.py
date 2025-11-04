# from sympy import O
import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import json
# import torchgeometry as tgy
from dld.data.render_joints.smplfk import set_on_ground_139, SMPLX_Skeleton

import sys
sys.path.insert(0,'.')
# from utils.parser_util import args

SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

SMPLX_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13,
                        15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
SMPLX_POSE_FLIP_PERM = []
for i in SMPLX_JOINTS_FLIP_PERM:
    SMPLX_POSE_FLIP_PERM.append(3*i)
    SMPLX_POSE_FLIP_PERM.append(3*i+1)
    SMPLX_POSE_FLIP_PERM.append(3*i+2)

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPLX parameters.
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose


Genres_Motorica = {
    # Hiphop	gLH	2	84
    # Krumping	gKR	1	18
    # Popping	gPO	2	42
    # Locking	gLO	2	18
    # Jazz	gJZ	2	52
    # Charleston	gCH	2	50
    # Tapping	gTP	2	11
    # Casual	gCA
    'gLH': 0,
    'gKR': 1,
    'gPO': 2,
    'gLO': 3,
    'gJZ': 4,
    'gCH': 5,
    'gTP': 6,
    'gCA': 7,
}



def music2genre(file_dir):
    music_genre = {}
    for file in os.listdir(file_dir):
        name = file.split(".")[0]
        genre = name.split('_')[1]
        if genre in Genres_Motorica:
            music_genre[name] = genre
        else:
            print("genre error!", genre)
        
    return music_genre

class Motorica_Smpl(data.Dataset):
    def __init__(self, args, istrain, dataname=None):
        self.motion_dir = eval(f"args.DATASET.{dataname.upper()}.MOTION")     
        self.music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC")  
        self.music2genre = music2genre(eval(f"args.DATASET.{dataname.upper()}.MOTION"))

        self.istrain = istrain
        self.args = args
        self.seq_len = args.FINEDANCE.full_seq_len

        self.motion_index = []
        self.music_index = []

        
        ignore_list, train_list, test_list = self.get_train_test_list(dataset = dataname)

        for name in ignore_list:
            if name in train_list:
                train_list.remove(name)
            if name in test_list:
                test_list.remove(name)

        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list         


        # for name in tqdm(self.datalist):
        #     name = name + ".npy"
        #     if name[:-4] in ignor_list:
        #         continue
        #     motion = np.load(os.path.join(self.motion_dir, name))
        #     music = np.load(os.path.join(self.music_dir, name))

        #     min_all_len = min(motion.shape[0], music.shape[0])
        #     motion = motion[:min_all_len]
          
        #     # hand smpl motion shape
        #     if motion.shape[-1] == 151 and args.FINEDANCE.nfeats ==139:
        #         motion = motion[:,:139]
        #     elif motion.shape[-1] == 151: # 151 dims: contacts(4) + root pos(3) + local 6D(24*6=144)
        #         pass
        #     else:
        #         print("motion.shape", motion.shape)
        #         raise("input motion shape error!")
        #     # sanity check if the music and motion are aligned
        #     assert motion.shape[0] == music.shape[0], f"motion and music length not equal for {name}! {motion.shape[0]} vs {music.shape[0]}"             

        #     genre_name = name.split('_')[1]
        #     genre_id = np.array(Genres_Motorica[genre_name])
        #     genre = torch.from_numpy(genre_id).unsqueeze(0)

        #     if args.FINEDANCE.mix:
        #         raise NotImplementedError("mix not implemented for Motorica yet")

        #     motion_all.append(motion)
        #     music_all.append(music)
        #     genre_list.append(genre)

        # self.motion_list = [m.astype(np.float32) for m in motion_all]
        # self.music_list = [m.astype(np.float32) for m in music_all]
        # self.genre_list = genre_list

        # self.len = len(self.motion_list)
        # print(f'Motorica has {self.len} samples..')

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        name = self.datalist[index] + ".npy"
        motion = np.load(os.path.join(self.motion_dir, name))
        music = np.load(os.path.join(self.music_dir, name))

        min_all_len = min(motion.shape[0], music.shape[0])
        motion = motion[:min_all_len]
        
        # hand smpl motion shape
        assert motion.shape[-1] == 151, "input motion shape error!"
        # sanity check if the music and motion are aligned
        assert motion.shape[0] == music.shape[0], f"motion and music length not equal for {name}! {motion.shape[0]} vs {music.shape[0]}"             

        genre_name = name.split('_')[1]
        genre_id = np.array(Genres_Motorica[genre_name])
        genre = torch.from_numpy(genre_id).unsqueeze(0)

        return motion, music, genre
    
    def get_train_test_list(self, dataset="FineDance"):
        if dataset in ["AISTPP", "AISTPP_60FPS"]:
            train = []
            test = []
            ignore = []

            train_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_train.txt', 'r')
            for fname in train_file.readlines():
                train.append(fname.strip())
            train_file.close()

            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_test.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()
                              
            test_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/splits/crossmodal_val.txt', 'r')
            for fname in test_file.readlines():
                test.append(fname.strip())
            test_file.close()

            ignore_file = open('/data2/lrh/dataset/aist/data/origin/aist_plusplus_final/ignore_list.txt', 'r')
            for fname in ignore_file.readlines():
                ignore.append(fname.strip())
            ignore_file.close()

            return ignore, train, test

        elif dataset == "AISTPP_LONG263":
            train = []
            test = []
            ignore = []
            print("modir", self.motion_dir)
            for file in os.listdir(self.motion_dir):
                if file[-4:] != '.npy':
                    continue
                file = file.split('.')[0]
                if file.split('_')[-1] in ['mLH5', 'mJS4', 'mBR3', 'mMH2', 'mPO1', 'mWA0']:
                    test.append(file)
                else:
                    train.append(file)

            return  ignore, train, test


        elif dataset == "FINEDANCE":
            all_list = []
            train_list = []
            for i in range(1,212):
                all_list.append(str(i).zfill(3))
    
            test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
            ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
            tradition_list = ['005', '007', '008', '015', '017', '018', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '032', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '126', '127', '132', '133', '134',  '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '151', '152', '153', '154', '155', '170']
            morden_list = []
            for one in all_list:
                if one not in tradition_list:
                    morden_list.append(one)

            ignor_list = ignor_list
            for one in all_list:
                if one not in test_list:
                    train_list.append(one)
            
            if self.args.FINEDANCE.partial == 'full':
                return ignor_list, train_list, test_list
            elif self.args.FINEDANCE.partial == 'morden':
                for one in train_list:
                    if one in tradition_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in tradition_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list
            elif self.args.FINEDANCE.partial == 'tradition':
                for one in train_list:
                    if one in morden_list:
                        train_list.remove(one)
                for one in test_list:
                    if one in morden_list:
                        test_list.remove(one)
                return ignor_list, train_list, test_list
        
        elif dataset == "motorica":
            train_list = []
            test_list = []
            ignor_list = []
            test_keys = [ "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006",
                "kthjazz_gJZ_sFM_cAll_d02_mJZ_ch01_bennygoodmansugarfootstomp_003",
                "kthjazz_gTP_sFM_sngl_d02_015",
                "kthmisc_gCA_sFM_cAll_d01_mCA_ch24",
                "kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001",
                "kthstreet_gLH_sFM_cAll_d01_mLH_ch01_thisisit_001",
                "kthstreet_gLH_sFM_cAll_d02_mLH_ch01_lala_001",
                "kthstreet_gLO_sFM_cAll_d02_mLO_ch01_arethafranklinrocksteady_002",
                "kthstreet_gPO_sFM_cAll_d01_mPO_ch01_bombom_002",
                "kthstreet_gPO_sFM_cAll_d02_mPO_ch01_bombom_001"]
            
            for file in os.listdir(self.motion_dir):
                file_stem = file.split(".")[0]
                # if any of the test keys is in the file_stem, then it is test file
                if any(test_key in file_stem for test_key in test_keys):
                    test_list.append(file_stem)
                else:
                    train_list.append(file_stem)
            
            print(f'train num: {len(train_list)}, test num: {len(test_list)}')

            return ignor_list, train_list, test_list
        
        else:
            raise ValueError(f"dataset name error! {dataset}")






if __name__ == '__main__':
    print('done')