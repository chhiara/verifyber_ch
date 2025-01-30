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
                 labels_name,
                 data_name,
                 #k=4,
                 same_size=False,
                 act=True,
                 fold_size=None,
                 transform=None,
                 distance=None,
                 self_loops=None,
                 with_gt=True,
                 return_edges=False,
                 split_obj=False,
                 train=True,
                 load_one_full_subj=False,
                 standardize=False,
                 centering=False,
                 labels_dir=None,
                 permute=False,
                 data_ext='npy',#data extension can be 'npy' or 'trk'):
                 bundle_name=""): 
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
        #self.k = k
        self.transform = transform
        self.distance = distance
        self.fold_size = fold_size
        self.act = act
        self.self_loops = self_loops
        self.with_gt = with_gt
        self.return_edges = return_edges
        self.fold = []
        self.n_fold = 0
        self.train = train
        self.load_one_full_subj = load_one_full_subj
        self.same_size = same_size
        self.standardize = standardize
        self.centering = centering
        self.permute = permute
        self.data_ext = data_ext
        self.data_name = data_name
        self.bundle_name= bundle_name
        if fold_size is not None:
            self.load_fold()
        if train:
            split_obj=False
        if split_obj:
            self.remaining = [[] for _ in range(len(subjects))]
        
        self.split_obj = split_obj
        
        assert self.data_ext == "npy" or self.data_ext == "trk", f"data_ext: {self.data_ext}"
        assert os.path.isdir(self.root_dir )

        if with_gt:
            self.labels = []
            for sub in subjects:
                sb_tr, sub_br=sub.split("__")
                pattern_lab_fn=os.path.join(self.root_dir, sb_tr, sb_tr + '*' + sub_br + '*' + labels_name + '*'+ bundle_name + '*' )
                label_file_li=glob.glob(pattern_lab_fn)
                assert len(label_file_li)==1, f"Error: Number of Label file unexpected, should be 1 but N files:{len(label_file_li)}\n {label_file_li}\n pattern_lab_fn: {pattern_lab_fn}"
                print(f"pattern_lab_fn", pattern_lab_fn)
                
                label_file = label_file_li[0]
                
                if label_file[-4:] == '.txt':
                    self.labels.append(np.loadtxt(label_file))
                elif label_file[-4:] == '.npy':
                    self.labels.append(np.load(label_file))
                else:
                    raise(f"Error, label file extension not recognized. Should be .txt or .npy")

            assert len(self.labels) == len(subjects)
            """
            #hcp version
            for sub in subjects:
                label_sub_dir = os.path.join(self.root_dir.rsplit('/',1)[0], labels_dir ,'sub-%s' % sub)
                label_file = os.path.join(label_sub_dir, 'sub-%s_var-HCP_labels_gt20mm.npy' % (sub))
                #label_file = os.path.join(label_sub_dir, 'sub-%s_CSD5TT8_weight.npy' % (sub))
                if label_file[-4:] == '.txt':
                    self.labels.append(np.loadtxt(label_file))
                else:
                    self.labels.append(np.load(label_file))
            """

    def __len__(self):
        if self.load_one_full_subj:
            return len(self.full_subj[0])
        return len(self.subjects)

    def __getitem__(self, idx):
        fs = self.fold_size
        if fs is None:
            #print(self.subjects[idx])
            #t0 = time.time()
            item = self.getitem(idx)
            #print('get item time: {}'.format(time.time()-t0))
            return item

        fs_0 = (self.n_fold * fs)
        idx = fs_0 + (idx % fs)

        return self.data_fold[idx]


    def load_fold(self):
        print('loading fold')
        fs = self.fold_size
        fs_0 = self.n_fold * fs
        #t0 = time.time()
        print('Loading fold')
        self.data_fold = [self.getitem(i) for i in range(fs_0, fs_0 + fs)]
        #print('time needed: %f' % (time.time()-t0))

    def getitem(self, idx):
        """
        Return sample
        sample = {  'points': 
                        xx #list of streamlines idx sampled ---at the bginning of the fct--
                           #then it becames the graph object used in input by pytorch geometric. 
                           See below

                    'gt': 
                            xx  #ground truth of the streamlines. np array of 0,1 if 2 classes
                    'obj_idxs':
                            #list of streamlines idx sampled (defined only if self.split_obj==True)
                    'obj_full_size':
                            n_streamlines total in the subject (defined only if self.split_obj==True)
                    'name':
                            filename of the tractogram
            
                    'dir':  self.root_dir
                        
                        }
        """
        sub = self.subjects[idx]

        sb_tr, sub_br=sub.split("__")
        pattern_data_fn=os.path.join(self.root_dir, sb_tr, sb_tr + '*' + sub_br + '*' +  self.data_name + '*'+ self.bundle_name + '*' + self.data_ext)
        #pattern_data_fn=os.path.join(self.root_dir, '*' + sub + '*' + self.data_name + '*' + self.data_ext)
        T_file_li=glob.glob(os.path.join(pattern_data_fn))
        print(f"pattern_data_fn", pattern_data_fn)
        assert len(T_file_li)==1, f"Error: Number of Data file unexpected, should be 1 but N files: {len(T_file_li)}\n Files: {T_file_li}\n pattern_data_fn: {pattern_data_fn}"
        T_file= T_file_li[0]

        #version hcp
        #sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        #T_file = os.path.join(sub_dir, 'sub-%s_var-HCP_full_tract_gt20mm.trk' % (sub))
        #T = nib.streamlines.load(T_file, lazy_load=True)
        
        if self.data_ext=="trk":
            T = nib.streamlines.load(T_file, lazy_load=True)
            n_streamlines=T.header['nb_streamlines']

        elif self.data_ext=="npy":
            streams_laoded = np.load(T_file, allow_pickle=True)
            n_streamlines = streams_laoded.shape[0]
        else:
            raise(f"Error: data_ext nor 'npy or 'trk' as expected, but {self.data_ext}")


        gt = self.labels[idx]
        #print('\tload gt %f' % (time.time()-t0))
        if self.split_obj:
            #split_obj: is True only at test time. If true each subject is loaded 
            #in different batches, until the streamlines are all loaded. So I have to consder which streamlines
            #are still not loaded in each iteration. This is is stored in self.remaining[idx] 
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(n_streamlines))
            sample = {'points': np.array(list(self.remaining[idx]))}
            if self.with_gt:
                sample['gt'] = gt[list(self.remaining[idx])]
        else:
            sample = {'points': np.arange(n_streamlines), 'gt': gt}
        #print(sample['name'])

        #t0 = time.time()
        if self.transform:
            #sample randomly some streamlines in a number equal to fixed size
            #in this function I sample n streamlines randomly from all available streamlines
            #sample is transfrmed so that 
            #       sample={points: idx of streamlines sampled of len fixed_size,
                            #gt: are the corresponding ground truth}

            sample = self.transform(sample)
        #print('\ttime sampling %f' % (time.time()-t0))

        if self.split_obj:
            #  self.remaining[idx] is updated to remove the sampled streamlines
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = n_streamlines
            #sample['streamlines'] = T.streamlines

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        
        sample['dir'] = self.root_dir
        #sample['dir'] = sub_dir #hcp version

        #number of streamlines (of indexes of streamlines after random sampling)
        #sample[points]: indexes of streamlines sampled after random sampling in self.transform
        n = len(sample['points'])

        #t0 = time.time()

        #self.same_size: I (chhiara) think that same_size==True if I am 
        #assuming the streamlines have each one the same number of points
        #False otherwise
        
        #new code
        if self.data_ext=="trk":
            #NB: by defualt load streamlineuses array:flat
            streams, lengths = load_streamlines_fast(T_file,
                                                    sample['points'].tolist())
            #print(f"streams.shape: {streams.shape}")
            #print(f"sample['points']: {len(sample['points'])}")

        elif self.data_ext=="npy":
            
            
            #streams_sampled = np.array([streams_laoded[i_streamline] for i_streamline in sample['points']], object)
            streams_sampled = streams_laoded[sample['points']] 
            lengths = np.array([ s.shape[0] for s in streams_sampled], int)
            streams = (np.concatenate(streams_sampled, axis=0)).astype(np.float32)
            
            #print(f"streams_sampled: {streams_sampled.shape}")
            #print(f"streams_sampled[0].shape: {streams_sampled[0].shape}")
            #print(f"streams.shape: {streams.shape}")
            #print(f"sample['points']: {len(sample['points'])} {type(sample['points'])}   sample['points'][:4]{sample['points'][:4]}")
            

        if self.same_size:
            #present in hcpc version
            #streams, lengths = load_streamlines_fast(T_file,
            #                                        sample['points'].tolist())
            if self.centering:
                #to center the coordiantes to zero
                streams_centered = streams.reshape(-1, lengths[0], 3)
                streams_centered -= streams_centered.mean(axis=1)[:,None,:]
                streams = streams_centered.reshape(-1,3)
            if self.permute:
                # import ipdb; ipdb.set_trace()
                #to flip the streamline or permute points order within streamlines. The latter makes sense only 
                #in case there are no edges between points
                streams_perm = self.permute_pts(
                    streams.reshape(-1, lengths[0], 3), type='flip')
                streams = streams_perm.reshape(-1, 3)
        else:
            #present in hcpc version
            #streams, lengths = load_streamlines_fast(T_file,
            #                                        sample['points'].tolist())

            if self.centering:
                streams_centered = self.center_sl_list(
                    np.split(streams, np.cumsum(lengths))[:-1])
                streams = np.vstack(streams_centered)

            if self.permute:
                streams_perm = self.permute_pts(
                    np.split(streams, np.cumsum(lengths))[:-1], type='flip')
                streams = streams_perm.reshape(-1, 3)
        
        if self.standardize:
            #stats_file = glob.glob(sub_dir + '/*_stats.npy')[0] #hcp version
            stats_file = glob.glob(self.root_dir + '/*_stats.npy')[0]
            mu, sigma, M, m = np.load(stats_file)
            streams = (streams - mu) / sigma


        #sample['points'] until now was the idx of the streamlines I am using
        #after running self.build_graph_sample 
        #sample['points'] is defined by:
            #  graph_sample: gData function: 
                                    #graph with as spatial position and 
                                    #feature of the nodes the streamlines points coordinates
                                    #lengths: number of streamlines points 
                                    #bvec: batch_vec
                                        #if lengths=[4,5,3]  ->  batch_vec=[0000,11111,222]
                            #graph_sample['edge_index'] = edges  -> defining the endges
                            #graph_sample['edge_attr'] = edge_attr 
                                # -> defining the edge attributes. not used so all def 1


        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    torch.from_numpy(sample['gt']) if self.with_gt else None)

       
        return sample

    def center_sl_list(self, sl_list):
        centers = np.array(map(functools.partial(np.mean, axis=0), sl_list))
        return map(np.subtract, sl_list, centers)

    def permute_pts(self, sl_list, type='rand'):
        perm_sl_list = []
        for sl in sl_list:
            if type == 'flip':
                perm_sl_list.append(sl[::-1])
            else:
                perm_idx = torch.randperm(len(sl)).tolist()
                perm_sl_list.append(sl[perm_idx])
        return np.array(perm_sl_list)


    def build_graph_sample(self, streams, lengths, gt=None):
        #t0 = time.time()
        #print('time numpy split %f' % (time.time()-t0))
        ### create graph structure
        #sls_lengths = torch.from_numpy(sls_lengths)
        lengths = torch.from_numpy(lengths).long()
        #print('sls lengths:',sls_lengths)

        #if lengths=[4,5,3]  ->  batch_vec=[0000,11111,222]
        #lengths: number of points of each streamline
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)

        #batch_slices: indexes of start of each streamline
        #first streamline begin from 0, the successive with the indexes defined by
        # the cumulative sum of the lengths
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])

        #erase first and last element
        slices = batch_slices[1:-1]


        streams = torch.from_numpy(streams)

        #number of points in total of all streamlines
        l = streams.shape[0]

        #features and spatial position correspond
        #x=streams -> features of the nodes
        #pos=streams-> spatial position of the nodes 
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             #sls_lengths=sls_lengths,
                             bvec=batch_vec,
                             pos=streams)
        #                     bslices=batch_slices)
        #edges = torch.empty((2, 2*l - 2*n), dtype=torch.long)


        if self.return_edges:
            #create the edges that should directionally connect the 
            #the contigous points, exclusind the terminal points of each streamline

            #erase the point index of the each last streamline point
            e1 = set(np.arange(0,l-1)) - set(slices.numpy()-1)

            #erase the point index of each first streamline point
            e2 = set(np.arange(1,l)) - set(slices.numpy())
            
            edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],
                            dtype=torch.long)
            
            
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr

        if self.distance:
            graph_sample = self.distance(graph_sample)
        #if self.self_loops:
        #graph_sample = self.self_loops(graph_sample)
        if gt is not None:
            graph_sample['y'] = gt

        return graph_sample


"""
#class crated from BIDS- does not work
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
        
        # Args:
        #     root_dir (string): root directory of the dataset.
        #     transform (callable, optional): Optional transform to be applied
        #         on a sample.
        
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
        
        # for sub in subjects:
        #     sub_dir = os.path.join(self.root_dir, sub)
        #     self.T_files.append(glob.glob(os.path.join(sub_dir, '*' + data_name + '*'))[0])
        
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
            
            # for sub in subjects:
            #     label_sub_dir = os.path.join(labels_dir, sub)
            #     label_file = glob.glob(os.path.join(label_sub_dir, '*' + labels_name + '*'))[0]
            #     if label_file[-4:] == '.txt':
            #         self.labels.append(np.loadtxt(label_file))
            #     else:
            #         self.labels.append(np.load(label_file))
            
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

    
    # #pietro version with possible inputs only trk file 
    # def getitem(self, idx):
    #     T_file = self.T_files[idx]
    #     T = nib.streamlines.load(T_file, lazy_load=True)

    #     if self.split_obj:
    #         if len(self.remaining[idx]) == 0:
    #             self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))
    #         stream_idxs = np.array(list(self.remaining[idx]))
    #     else:
    #         stream_idxs = np.arange(T.header['nb_streamlines'])

    #     sample = {'points': stream_idxs}
    #     if self.with_gt:
    #         sample['gt'] = self.labels[idx][stream_idxs]

    #     # this transform sample streamlines from subjects
    #     if self.transform:
    #         sample = self.transform(sample)

    #     if self.split_obj:
    #         self.remaining[idx] -= set(sample['points'])
    #         sample['obj_idxs'] = sample['points'].copy()
    #         sample['obj_full_size'] = T.header['nb_streamlines']

    #     sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
    #     sample['dir'] = T_file.rsplit('/', 1)[0]

    #     streams, lengths = load_streamlines_fast(T_file,
    #                                     sample['points'].tolist(),
    #                                     container='array_flat')
        
    #     if self.permute:
    #         streams_perm = self.permute_pts(
    #             np.split(streams, np.cumsum(lengths))[:-1], type=self.permute_type)
    #         streams = streams_perm.reshape(-1, 3)

    #     sample['points'] = self.build_graph_sample(streams,
    #                 lengths,
    #                 torch.from_numpy(sample['gt']) if self.with_gt else None)
    #     return sample
    
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
"""