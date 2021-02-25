# Module and Data package (MADpack)

> :warning: This package is at a very early stage, use it carefully and double check if everything works as expected!

MADpack is a collection of modules and datasets for efficiently working with pytorch. 
It is intended to extend the PyTorch and torchvision packages and facilitate common worksflows.

- Models and building blocks for models (all PyTorch-based)
- Datasets and extended base class (with automatic download-from-repository capability)
- Multiple data transformations to facilitate building datasets
- Straightforward data visualization using `plot_data(<dataset>)`

Learn how to use MADpack in the notebook `doctests/documentation.ipynb`.


## Installation

``python -m pip install git+https://github.com/timojl/madpack`

## Models

The provided models are organized in different categories:

* `resnet_dense`: Models for dense prediction, e.g. segmentation.
* `resnet_generalized`: Modified versions of PyTorch ResNet, e.g. less pooling or more narrow.
* `video`: Baselines for video models. E.g. 3D-convolutions and frame pooling.
* `functions`: torch functions to be used when no parameters need to be stored
* `modules`: torch nn.Modules when parameters need to be stored


## Datasets

### Base class
Baseclass for datasets with a some convenience functions over PyTorch's Dataset class.
It differentiates between the `DATASETS_PATH` (default `~/datasets`) where datasets are
stored locally and `REPOSITORY_PATH` (default `~/dataset_repositories`) where dataset archives 
are stored (possibly a network location). 

Normally each dataset's `__getitem__` method returns a tuple (inputs, targets) where inputs and
targets are tuples again.

- Automatically load the dataset from a repository (e.g. `agecker/datasets`).
- Check for overlaps between splits (`sample_ids` must be implemented)
- Resize the dataset dynamically using `resize()`.

### Datasets

Synthetic 
- `cABC`
- `Pathfinder`
- `CutteredOmniglot`
- `MNIST`
  
Natural Images
- `CUB200`
- `FSS1000`
- `ILSVRC2012 (ImageNet): 1000 categories with 1300 images each.`
- `Raven`

## Transformations
A collection of potentially helpful data processing functions can be found in `madpack.transforms`.


## Visualization

The idea is to feed everything to `plot_data` and 


## TODOs
- Convert functions in sampling from Numpy to PyTorch



