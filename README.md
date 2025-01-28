# Verifyber
**A supervised tractogram filtering method based on Geometric Deep Learning**. It allows the filtering out of non-plausible streamlines from tractograms

## Tractogram filtering script

The executable script is `tractogram_filtering.py`. It reads the configuration file, `run_config.json` to get arguments from "outside", and based on it performs different steps.

The script generates a temporary folder `TEMP=tmp_tractogram_filtering/`, where it stores in the subdirectories `TEMP/input/` and `TEMP/output/` the actual input and output files. Some intermediate files generated during the pre-processing steps are stored directly in the `TEMP` folder 

The input file is always a tractogram .trk, projected into MNI space with fixed number of points per streamline. 

The output are two text files containing the indexes of plausible and non-plausible fibers, and optionally the .trk of the filtered tractogram.    

## Configuration file
`run_config.json` is composed as follows:
- `trk`: path to the tractogram uploaded by the user
- `t1`: path to the T1w image in subject space. The image is preferred if it is a brain extracted image. In case no t1 or fa image is provided, the tractogram is assumed to be already in MNI space.
- `fa`: path to the FA image in subject space. The image is preferred if it is a brain extracted image. In case no t1 or fa image is provided, the tractogram is assumed to be already in MNI space.
- `resample_points`: T/F flag. If T the streamlines will be resampled to 16 points, otherwise no.
- `return_trk`: T/F flag. If T the filtered trk tractogram will be returned along with the indexes of plausible and non-plausible streamlines.
- `task`: classification/regression. [not used right now]
- `warp`: choices (lin | fast | slow). Type of co-registration to the standard using ANTs normalization tool. "lin" is a affine registration; "fast" (SUGGESTED) is a quick non-linear diffeormophic registration; "slow" is a more accurate non-linear diffeormophic registratio, requiring more time to compute.
- `model`: defalut = "sdec_extractor", choices are the names of the folder present in checkpoints/

## Usage
1. setup your env follwing instructions in [verifyber_updated_env.txt](verifyber_updated_env.txt)
2. run `tractogram_filtering.py -config <run_config.json>`

## Docker containers 
See docker://pietroastolfi/tractogram-filtering:<tag>, <tag>=cpu|gpu. Note that the gpu container works with CUDA 10


## Model notes
* DEC: 1st layer dinamic endge convolution - build by it self the graph structure
* sDEC: 1st layer edge convolution  - Samuel uses this

    *  2nd layer: dinamic edge convolution in both
    *  mathematically they should do the same thing



%## parameters documentation (partial)

## Set cuda device
* Define variable  device="cuda:<GPU-IDX>"  eg: device="cuda:0"       

* if I run  verifyber with a container, by passing GPU indexes, the indexing of the GPU inside the container starts always from 0. 

* In the code do: model.to(device)   and also laod the data to device


