# Overview


This repository is associated with the forthcoming paper "Pierson, Matthew., and Mehrabi, Zia. 2024. Mapping waterways worldwide with deep learning. arXiv.  	
https://doi.org/10.48550/arXiv.2412.00050". Please do cite this paper and attribute the work if using the model or work. The data outputs of this model (raster and vectorized versions) are also stored and available from the following source: Pierson, Matthew., Mehrabi. Zia. 2024, WaterNet Outputs and Code, https://doi.org/10.7910/DVN/YY2XMG, Harvard Dataverse.

# Setup

This module has the functionality to download data, train a WaterNet model, and deploy a WaterNet model.

There are scripts to download the data needed to train the model, and to further generate the training/ validation/ test
datasets. 

This process has not been optimized, especially in terms of data storage. In many cases we save intermediate datasets,
which can be deleted manually after the process has completed (and the user has confirmed the integrity of the process).
This is especially necessary when downloading the Sentinel 2 data, as downloads can fail without
crashing the entire program.

We ran these processes on a single machine with multiple drives. Most all of the data paths
can be configured. To set custom drive locations, make a copy of
[configuration_files/path_configuration_template.yaml](./configuration_files/path_configuration_template.yaml)
titled "path_configuration.yaml" and saved in the configuration_files directory. Uncomment the directories you
would like to change the path too, and add the absolute path to the new directory.

For example if you want the sentinel_merged directory to be at /Drive1/sentinel_merged and sentinel_unmerged to be
at /Drive2/waterways_data/sentinel_raw, the lines in the parth_configuration.yaml file should be

```
sentinel_merged: /Drive1/sentinel_merged
sentinel_unmerged: /Drive2/waterways_data/sentinel_raw
```

## Installation

Installation requires pytorch and [WaterNet](https://github.com/Better-Planet-Laboratory/WaterNet). You may also want to install
[WaterNet Vectorize](https://github.com/Better-Planet-Laboratory/WaterNet_vectorize).

All code was prototyped using python 3.11.4 and pip 23.0.1.

A python environment and version handler such as [pyenv](https://github.com/pyenv/pyenv) should make those easy to obtain.

After getting your environment setup correctly, download this repository and use pip to install:

```
git clone https://github.com/Better-Planet-Laboratory/WaterNet_training_and_evaluation.git
cd WaterNet_training_and_evaluation
pip install .
```

or if you wish to edit to code

``
pip install -e .
``


# Data Download for Training

The [data download scripts](./scripts/data_downloads) in should be run in the following order:

1. [download_nhd_data.py](./scripts/data_downloads/download_nhd_data.py)
2. [download_sentinel_tile_shapefile.py](./scripts/data_downloads/download_sentinel_tile_shapefile.py)
3. [download_usa_data_for_training.py](./scripts/data_downloads/download_usa_data_for_training.py)
4. [generate_training_data.py](./scripts/data_downloads/generate_training_data.py)
5. [tdx_hydro_basin_downloads.py](./scripts/data_downloads/tdx_hydro_basin_downloads.py) (Required for the vectorization process).

If all of these scripts have run successfully then the [train waternet script](./scripts/training/train_waternet.py) can be run, assuming you have installed
the WaterNet module.


# Deployment

While the functionality for deployment exists in the module, it is up to the user to develop the code to run this on
their desired region. The module function that should be used is
[water.deployment_functions.deploy_on_polygon.deploy_on_polygon](./src/water/deployment_functions/deploy_on_polygon.py)

As a note, it is assumed that you have downloaded the appropriate sentinel data and elevation data for the polygon you
intend to run this on, and that that data is stored in the sentinel_merged and elevation directories respectively.

# Related Repositories


 * [WaterNet](https://github.com/Better-Planet-Laboratory/WaterNet)
 * [WaterNet Vectorize](https://github.com/Better-Planet-Laboratory/WaterNet_vectorize)

# Citations

```yaml
@misc{pierson2024mappingwaterwaysworldwidedeep,
      title={Mapping waterways worldwide with deep learning}, 
      author={Matthew Pierson and Zia Mehrabi},
      year={2024},
      eprint={2412.00050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00050}, 
}
```

```yaml
@data{DVN/YY2XMG_2024,
author = {Pierson, Matthew and Mehrabi, Zia},
publisher = {Harvard Dataverse},
title = {{WaterNet Outputs and Code}},
year = {2024},
version = {V1},
doi = {10.7910/DVN/YY2XMG},
url = {https://doi.org/10.7910/DVN/YY2XMG}
}
```
