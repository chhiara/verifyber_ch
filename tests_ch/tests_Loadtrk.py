import numpy as np

from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast


T_file="/raid/home/nilab/chiara/datasets/test_verifyber/trk_data/sub-1112__FAT_L__affined_tract.trk"
li_idx=[0,4,5]
#Q: what is lennghts?
streams, lengths = load_streamlines_fast(T_file,
                               li_idx,
                                container='array_flat')

print(f"stream {stream.shape}")
print(f"lengths {lengths.shape}")
print(f"lengths {lengths[:2]}")



