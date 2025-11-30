# CryoBoltz ❄️ ⚡

[[Paper](https://arxiv.org/abs/2506.04490)] [[Website](https://cryoboltz.cs.princeton.edu/)]

CryoBoltz is a method for fitting atomic structures into cryo-EM density maps of dynamic proteins. It is built on top of Boltz-1, a state-of-the-art structure prediction model for biomolecular complexes. Through a multi-stage guidance mechanism that modifies the Boltz diffusion trajectory at inference time, CryoBoltz recovers diverse conformations from input cryo-EM data.

## Installation
We recommend installing CryoBoltz in a clean conda environment -- first clone the git repository, and then use `pip` to install the package from the source code:
```
conda create --name cryoboltz python=3.10
conda activate cryoboltz
git clone https://github.com/ml-struct-bio/cryoboltz.git
cd cryoboltz
pip install -e .
```
Run the following to check that the installation was successful:
```
boltz predict --help
```

## Prepare Inputs

CryoBoltz requires the following inputs:

1. The sequence to be modeled, in FASTA or YAML format. See the Boltz [prediction instructions](https://github.com/ml-struct-bio/cryoboltz/blob/main/docs/prediction.md) for more details.

2. A cryo-EM density map, in MRC format. The prediction may be sped up by cropping background regions of the map first -- see the Example section below. It is also recommended to have an estimated map resolution, and choose a map threshold that excludes background noise.

3. An existing structure that is roughly aligned with the map, in CIF format. This structure aids in pre-aligning the CryoBoltz prediction to the map during the diffusion sampling process, but is not otherwise used as a template. This may be, e.g., an unguided Boltz prediction aligned to the map, or a partially built model from [ModelAngelo](https://github.com/3dem/model-angelo). Please ensure that the chain labels in the input structure match those of the sequence file.


## Run Prediction

Once the inputs have been prepared, prediction can be run with a single command:
```
boltz predict <SEQ_PATH> --density_map <MRC_PATH> --aligned_model <CIF_PATH> --res <RESOLUTION> --thresh <THRESHOLD> --use_msa_server --out_dir results
```
Below are the available arguments specific to CryoBoltz. For a full list of options of the base Boltz model, see its [prediction instructions](https://github.com/ml-struct-bio/cryoboltz/blob/main/docs/prediction.md).

| **Option**              | **Type**        | **Default**                 | **Description**                                                                                                                                                                      |
|-------------------------|-----------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--density_map`         | `path`          | `None`                  | The path to the MRC density map. Required.                                                                     |
| `--aligned_model`               | `path`          | `None`          | The path to the aligned structure CIF. Required.
| `--res`               | `float`          | `2.0`          | Estimated map resolution.
| `--thresh`               | `float`          | `0.0`          | Voxels below this value will be zeroed during global guidance. Set to remove background noise.   
| `--dust`               | `int`          | `5`          | Small map blobs (connected components) below this size will be removed during global guidance. Set to remove background noise.
| `--global_steps`               | `int int`          | `101 150`          | 1-indexed diffusion timesteps between which global guidance is active.
| `--global_scale`               | `float float`          | `0.25 0.05`          | Guidance strength at beginning and end of the global guidance phase.
| `--no_global`               | `flag`          | `False`          | Whether to disable global guidance.
| `--local_steps`               | `int int`          | `151 175`          | 1-indexed diffusion timesteps between which local guidance is active.
| `--local_scale`               | `float float`          | `0.5 0.5`          | Guidance strength at beginning and end of the local guidance phase.
| `--no_local`               | `flag`          | `False`          | Whether to disable local guidance.
| `--cloud_size`               | `float`          | `0.25`          | Scaling factor for point cloud size during global guidance.
| `--voxel_batch`               | `int`          | `32768`          | Batch size of voxels to process concurrently during local guidance. Reduce this value if an out-of-memory error occurs during local guidance.
| `--write_traj`               | `flag`          | `False`          | Whether to write out all intermediate diffusion structures as XTC trajectories.
| `--write_guidance_loss`               | `flag`          | `False`          | Whether to write out guidance loss curves to an npz file.

See the Boltz [prediction instructions](https://github.com/ml-struct-bio/cryoboltz/blob/main/docs/prediction.md) for a description of all output files.

## Example

Here we walk through the modeling of a Pma1 monomer in its autoinhibited state ([PDB:9UGC](https://www.rcsb.org/structure/9UGC)).

### Preparing the sequence
Download the FASTA file from [PDB:9UGC](https://www.rcsb.org/structure/9UGC). Following the Boltz formatting convention, and restricting ourselves to a single monomer, we rewrite the header. The file (*rcsb_pdb_9UGC.fasta*) should looks as follows:
```
>A|protein
MTDTSSSSSS...
```

### Preparing the map
Download and decompress the map file from [EMDB:64136](https://www.ebi.ac.uk/emdb/EMD-64136). We note that the reported resolution is **3.52 Å**. Furthermore, by visual inspection (e.g. in [ChimeraX](https://www.rbvi.ucsf.edu/chimerax/)), we choose **0.35** as the map threshold. The prediction can be sped up by cropping out background regions of the map. Here, we crop to the density at the 0.35 threshold with 10 voxels of padding on each side:
```
python scripts/preproc_map.py emd_64136.map --thresh 0.35 --pad 10 -o map_cropped.mrc
```
Alternatively, the `--dim` option can be used to crop the map to a fixed size.

### Preparing the input structure
For the input structure, we will use an unguided Boltz prediction that has been aligned to the map. We have provided such a structure in *examples/input_struct.cif*. To create your own, run Boltz without cryo-EM inputs:
```
boltz predict rcsb_pdb_9UGC.fasta --diffusion_samples 1 --use_msa_server --out_dir unguided_results
```
Then align the prediction (*unguided_results/boltz_results_rcsb_pdb_9UGC/predictions/rcsb_pdb_9UGC/rcsb_pdb_9UGC_model_0.cif*) to the map, e.g. using the `fitmap` function in [ChimeraX](https://www.rbvi.ucsf.edu/chimerax/).

### Running the prediction
With the inputs ready, we can now run Boltz with map guidance:
```
boltz predict rcsb_pdb_9UGC.fasta --diffusion_samples 5 --use_msa_server --density_map map_cropped.mrc --aligned_model examples/input_struct.cif --res 3.52 --thresh 0.35 --out_dir cryoboltz_results
```
The predicted structures are ordered by decreasing confidence score. We suggest visually assessing the fit of each candidate structure to the map. To select candidates based on measures of map-model fit, we refer to the tools available in [Phenix](https://phenix-online.org/documentation/reference/validation_cryo_em.html) and other validation packages.

<!-- ### Ranking the output structures
By default, the predicted structures are ordered by decreasing confidence score. We suggest visually assessing the fit of each candidate structure to the map. We also provide a script for ranking the structures by map-model fit. It computes the real space correlation coefficient (RSCC) using a simple forward measurement model. For more comprehensive fit metrics, we refer to the tools available in [Phenix](https://phenix-online.org/documentation/reference/validation_cryo_em.html).
```
python scripts/rank.py cryoboltz_results
```
This generates a CSV file in the same directory as the output CIF files. -->

## Contact
For any feedback, questions, or bugs, please open a Github issue.

## Cite

```bibtex
@article{raghu2025cryoboltz,
  title={Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data},
  author={Raghu, Rishwanth and Levy, Axel and Wetzstein, Gordon and Zhong, Ellen D.},
  journal={Advances in neural information processing systems},
  year={2025},
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```
