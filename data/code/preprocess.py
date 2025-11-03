import argparse
import os
from pydoc import doc
from cv2 import mean
import numpy as np
from pathlib import Path
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import wavfile
sys.path.append(os.getcwd()) 
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk, ax_to_6v, ax_from_6v, SMPLSkeleton, do_smplfk
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle)
import librosa
floor_height = 0




def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def set_on_ground(root_pos, local_q_axis, human_model):
    # if human_model is SMPL
    l_toe_index = 10
    r_toe_index = 11
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = human_model.forward(local_q_axis, root_pos)
    positions = torch.reshape(positions, (length, -1, 3)) # bxt, j, 3

    l_toe_h = positions[0, l_toe_index, 1] - floor_height
    r_toe_h = positions[0, r_toe_index, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos

def set_on_ground_139(data, smplx_model, ground_h=0):
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    positions = do_smplxfk(data, smplx_model)
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[:, 5] = data[:, 5] - (height -  ground_h)

    return data

def audio_feats_extract(wav_path):
    FPS = 30 #* 5
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    audio_name = Path(wav_path).stem
    # remove file extension
    audio_name = audio_name.split(".")[0]
    print(f'Processing audio: {audio_name}')
    
    data, _ = librosa.load(wav_path, sr=SR)

    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    print(f'envelope shape: {envelope.shape}')
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
    ).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)


    start_bpm = librosa.beat.tempo(y=librosa.load(wav_path)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )
    print(f'audio_feature shape: {audio_feature.shape}')
    exit()
    return audio_feature

def motion_feats_extract(moinputs_dir, mooutputs_dir, music_indir, music_outdir):
    raw_fps = 30
    data_fps = 30
    data_fps <= raw_fps
    device = "gpu"
    # smplx_model = SMPLX_Skeleton()
    # we use smpl model instead of smplx for motorica data
    smpl_model = SMPLSkeleton()

    os.makedirs(mooutputs_dir, exist_ok=True)
    os.makedirs(music_outdir, exist_ok=True)
        
    motions = sorted(glob.glob(os.path.join(moinputs_dir, "*.npy")))
    for motion in tqdm(motions):
        print(motion)
        data = np.load(motion, allow_pickle=True).item()
        motion_data = data["motion"]['motion_data']
        data_fps = data['current_fps']
        fname = os.path.basename(motion).split(".")[0].replace("_motion", "")
        # fname = data['motion']['file_name']
        wav_path = os.path.join(music_indir, fname + ".wav")
        # process music features
        music_fea = audio_feats_extract(wav_path)
        np.save(os.path.join(music_outdir, fname+".npy"), music_fea)


        print(f'motion data shape: {motion_data.shape}')

        # convert numpy arrays to torch tensors before calling pytorch3d / SMPL routines
        root_pos  = motion_data[:, 0, :3]   # (150, 3)
        local_q_mat_flattented = motion_data[:, 1:, :] #(150, 24, 9)
        # reshape into (T, 24, 3, 3)
        local_q_mat = local_q_mat_flattented.reshape(local_q_mat_flattented.shape[0], local_q_mat_flattented.shape[1], 3, 3)

        # pytorch3d expects torch tensors (not numpy). Convert and use matrix_to_axis_angle which returns a torch tensor.
        local_q_mat_t = torch.from_numpy(local_q_mat).float()
        local_q_axis_t = matrix_to_axis_angle(local_q_mat_t)  # (T, 24, 3) tensor
        # flattened axis-angle for FK: (T, 72)
        local_q_axis_flattened = local_q_axis_t.reshape(local_q_axis_t.shape[0], -1)

        # convert root_pos to torch tensor as well
        root_pos = torch.from_numpy(root_pos).float()
        length = root_pos.shape[0]
        # set on ground and run FK using torch tensors
        root_pos = set_on_ground(root_pos.unsqueeze(0), local_q_axis_t.unsqueeze(0), smpl_model)
        positions = smpl_model.forward(local_q_axis_t.unsqueeze(0), root_pos)
        positions = positions.view(length, -1, 3)   # bxt, j, 3

        # contacts

        feet = positions[:, (7, 8, 10, 11)]  # # 150, 4, 3
        contacts_d_ankle = (feet[:,:2,1] < 0.12).to(local_q_axis_flattened)
        contacts_d_teo = (feet[:,2:,1] < 0.05).to(local_q_axis_flattened)
        contacts_d = torch.cat([contacts_d_ankle, contacts_d_teo], dim=-1).detach().cpu().numpy()

        local_q_axis = local_q_axis_flattened.view(length, 24, 3)  
        local_q_6v_flattented = ax_to_6v(local_q_axis).view(length, 24*6).detach().cpu().numpy()
        # root_pos is a torch tensor here; convert to numpy for concatenation
        root_pos_np = root_pos.detach().cpu().numpy().squeeze(0)
        mofeats_input = np.concatenate([contacts_d, root_pos_np, local_q_6v_flattented], axis=-1)
        np.save(os.path.join(mooutputs_dir, fname+".npy"), mofeats_input)
    return


if __name__ == "__main__":
    # motion_feats_extract(#moinputs_dir='/data2/librosah/dataset/fine_dance/origin/motion_feature315', 
    #                     moinputs_dir='data/finedance/motion/', 
    #                     mooutputs_dir="data/finedance/mofea319/", 
    #                     music_indir="data/finedance/music_npy", 
    #                     # music_indir="/data2/librosah/dataset/fine_dance/origin/music_feature35_edge",
    #                     music_outdir="data/finedance/music_npynew/", )
    motion_feats_extract(
        moinputs_dir='/fs/nexus-projects/PhysicsFall/editable_dance_project/data/motorica/sliced_motion_smpl',
        mooutputs_dir='/fs/nexus-projects/PhysicsFall/LODGE/data/motorica/mofea319',
        music_indir='/fs/nexus-projects/PhysicsFall/editable_dance_project/data/motorica/sliced_audio',
        music_outdir='/fs/nexus-projects/PhysicsFall/LODGE/data/motorica/music_npynew',

    )
