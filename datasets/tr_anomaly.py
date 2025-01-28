from __future__ import print_function
import os
import os.path
import torch
import numpy as np

import nibabel as nib
import glob
from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset

class TractAnomlayDataset(gDataset):
    def __init__(self,
                 sub_file,
                 root_dir,
                 run='train',
                 data_name='.trk',
                 transform=None,
                 with_gt=True,
                 return_edges=False,
                 split_obj=False,
                 labels_dir=None,
                 labels_name='labels',
                 permute=False,
                 permute_type='flip',
                 data_ext='npy'): #data extension can be 'npy' or 'trk'
        """
        Args:
            root_dir (string): root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        with open(sub_file) as f:
            subjects = [s.strip() for s in f.readlines()]
        self.subjects = subjects
        self.transform = transform
        self.with_gt = with_gt
        self.return_edges = return_edges
        self.train = run == 'train'
        self.permute = permute
        self.permute_type = permute_type
        self.data_ext=data_ext

        assert self.data_ext == "npy" or self.data_ext == "trk", f"data_ext: {self.data_ext}"
        assert os.path.isdir(self.root_dir )

        if self.train:
            split_obj=False
            print("Training dataset creation")

        self.split_obj = split_obj

        self.T_files = []
        """
        for sub in subjects:
            sub_dir = os.path.join(self.root_dir, sub)
            self.T_files.append(glob.glob(os.path.join(sub_dir, '*' + data_name + '*'))[0])
        """
        for sub in subjects:
            #sub_dir = os.path.join(self.root_dir, sub)
            pattern_data_fn=os.path.join(self.root_dir, '*' + sub + '*' + data_name + '*' + data_ext)
            T_file_li=glob.glob(os.path.join(pattern_data_fn))
            assert len(T_file_li)==1, f"Error: Number of Data file unexpected, should be 1 but N files: {len(T_file_li)}\n Files: {T_file_li}\n pattern_data_fn: {pattern_data_fn}"
            
            self.T_files.append(T_file_li[0])


        assert len(self.T_files) == len(subjects)
        print(f"Found {len(self.T_files)} data for training")

        if with_gt:
            self.labels = []
            labels_dir = labels_dir if labels_dir is not None else root_dir
            """
            for sub in subjects:
                label_sub_dir = os.path.join(labels_dir, sub)
                label_file = glob.glob(os.path.join(label_sub_dir, '*' + labels_name + '*'))[0]
                if label_file[-4:] == '.txt':
                    self.labels.append(np.loadtxt(label_file))
                else:
                    self.labels.append(np.load(label_file))
            """
            labels_dir = labels_dir if labels_dir is not None else root_dir
            for sub in subjects:
                pattern_lab_fn=os.path.join(self.root_dir, '*' + sub + '*' + labels_name + '*')
                label_file_li=glob.glob(pattern_lab_fn)
                assert len(label_file_li)==1, f"Error: Number of Label file unexpected, should be 1 but N files:{len(label_file_li)}\n {label_file_li}\n pattern_lab_fn: {pattern_lab_fn}"
                
                label_file = label_file_li[0]

                #print(label_file)
                
                if label_file[-4:] == '.txt':
                    self.labels.append(np.loadtxt(label_file))
                elif label_file[-4:] == '.npy':
                    self.labels.append(np.load(label_file))
                else:
                    raise(f"Error, label file extension not recognized. Should be .txt or .npy")

            assert len(self.labels) == len(subjects) and len(self.labels) == len(self.T_files)
            #print(self.T_files)   
                
        
        if split_obj:
            self.remaining = [[] for _ in range(len(subjects))]


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item


    def getitem(self, idx):
        #-----my version of getitem that can take in input npy or trk

        T_file = self.T_files[idx]
        if self.data_ext=="trk":
            T = nib.streamlines.load(T_file, lazy_load=True)
            n_streamlines=T.header['nb_streamlines']

        elif self.data_ext=="npy":
            streams_sep = np.load(T_file, allow_pickle=True)
            n_streamlines = streams_sep.shape[0]
        else:
            raise(f"Error: data_ext nor 'npy or 'trk' as expected, but {self.data_ext}")

        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(n_streamlines))
            stream_idxs = np.array(list(self.remaining[idx]))
        else:
            stream_idxs = np.arange(n_streamlines)

        #ponits: streamlines idx to sample
        sample = {'points': stream_idxs}

        if self.with_gt:
            sample['gt'] = self.labels[idx][stream_idxs]

        # this transform sample streamlines from subjects
        if self.transform:
            sample = self.transform(sample)

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = n_streamlines

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = T_file.rsplit('/', 1)[0]


        if self.data_ext=="trk":
            streams, lengths = load_streamlines_fast(T_file,
                                            sample['points'].tolist(),
                                            container='array_flat')
        elif self.data_ext=="npy":
            lengths = np.array([ s.shape[0] for s in streams_sep], int)
            streams = np.concatenate(streams_sep, axis=0)
            streams = streams.astype(np.float32)



        
        if self.permute:
            #permute the streamlines by flipping them, or by randomly eprmuting the points
            streams_perm = self.permute_pts(
                np.split(streams, np.cumsum(lengths))[:-1], type=self.permute_type)
            streams = streams_perm.reshape(-1, 3)

        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    torch.from_numpy(sample['gt']) if self.with_gt else None)
        return sample

    """
    #pietro version with possible inputs only trk file 
    def getitem(self, idx):
        T_file = self.T_files[idx]
        T = nib.streamlines.load(T_file, lazy_load=True)

        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))
            stream_idxs = np.array(list(self.remaining[idx]))
        else:
            stream_idxs = np.arange(T.header['nb_streamlines'])

        sample = {'points': stream_idxs}
        if self.with_gt:
            sample['gt'] = self.labels[idx][stream_idxs]

        # this transform sample streamlines from subjects
        if self.transform:
            sample = self.transform(sample)

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = T_file.rsplit('/', 1)[0]

        streams, lengths = load_streamlines_fast(T_file,
                                        sample['points'].tolist(),
                                        container='array_flat')
        
        if self.permute:
            streams_perm = self.permute_pts(
                np.split(streams, np.cumsum(lengths))[:-1], type=self.permute_type)
            streams = streams_perm.reshape(-1, 3)

        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    torch.from_numpy(sample['gt']) if self.with_gt else None)
        return sample
    """
    def permute_pts(self, sl_list, type='rand'):
        perm_sl_list = []
        for sl in sl_list:
            if type == 'flip':
                #here we are flipping the streamlines: inverting the points order
                perm_sl_list.append(sl[::-1])
            else:
                #here we are randomly permute points within the same streamline
                perm_idx = torch.randperm(len(sl)).tolist()
                perm_sl_list.append(sl[perm_idx])
        return np.array(perm_sl_list)

    def build_graph_sample(self, streams, lengths, gt=None):
        ### create graph structure
        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             bvec=batch_vec,
                             pos=streams)
        if self.return_edges:
            e1 = set(np.arange(0,l-1)) - set(slices.numpy()-1)
            e2 = set(np.arange(1,l)) - set(slices.numpy())
            edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],
                            dtype=torch.long)
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr
        if gt is not None:
            graph_sample['y'] = gt

        return graph_sample
