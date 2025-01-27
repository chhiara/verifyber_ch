import numpy as np
import sys
import os

repo_path=os.environ["HOME"] + f"/local/verifyber_ch/"
sys.path.append(repo_path)

from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast


T_file="/home/chiara/mnt/hd14TB_alberto/chhiara/TractoInferno_Nilab_TractoAnomaly/derivatives/register_to_MNI_affine/sub-1112/sub-1112__FAT_L__affined_tract.trk"

li_idx=[0,4,5]
#Q: what is lennghts?
streams, lengths = load_streamlines_fast(T_file,
                               li_idx,
                                container='array_flat')

print(f"stream {stream.shape}")
print(f"lengths {lengths.shape}")
print(f"lengths {lengths[:2]}")



