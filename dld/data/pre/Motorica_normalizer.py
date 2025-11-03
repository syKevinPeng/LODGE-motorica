import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# make repo importable (same style as FineDance_normalizer)
sys.path.append(os.getcwd())
from dld.data.utils.preprocess import Normalizer

# motorica data folder (explicit path provided)
modir = '/fs/nexus-projects/PhysicsFall/LODGE/data/motorica/mofea319'

data_li = []
for fname in tqdm(os.listdir(modir), desc="Reading motorica files"):
	if not fname.endswith('.npy'):
		continue
	filepath = os.path.join(modir, fname)
	try:
		motion = np.load(filepath, allow_pickle=True)
	except Exception as e:
		print(f"Failed to load {filepath}: {e}")
		continue


	if motion is None:
		print(f"Could not find motion array in {filepath}, skipping.")
		continue

	# ensure 2D array (T, F)
	if motion.ndim == 1:
		motion = motion.reshape(-1, 1)

	# convert to float32 and to torch tensor
	try:
		tensor = torch.from_numpy(motion.astype(np.float32))
	except Exception as e:
		print(f"Failed to convert {filepath} to tensor: {e}")
		continue

	# append per-frame (1, F) slices (same strategy as FineDance_normalizer)
	for idx in range(tensor.shape[0]):
		data_li.append(tensor[idx].unsqueeze(0))

if len(data_li) == 0:
	raise RuntimeError(f"No frames collected from {modir} -- check path and file formats.")

data_li = torch.cat(data_li, dim=0)
data_li_ori = data_li.clone()

# build normalizer and save
Normalizer_ = Normalizer(data_li)
out_path = Path('/fs/nexus-projects/PhysicsFall/LODGE/data')
out_path.mkdir(parents=True, exist_ok=True)
save_file = out_path / 'Motorica_Normalizer.pth'
torch.save(Normalizer_, str(save_file))

# load back and quick sanity check
reNorm = torch.load(str(save_file))
data_newnormed = reNorm.normalize(data_li)
data_newunnormed = reNorm.unnormalize(data_newnormed)

print("Example normalized values:", data_newnormed[0, :20])
print("Recovered (unnormalized) values:", data_newunnormed[0, :20])
print("Original values:", data_li_ori[0, :20])
print(f"Saved normalizer to: {save_file.resolve()}")

