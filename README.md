# CtrTab
Official Implementations of "Towards Synthesizing High-Dimensional Tabular Data with Limited Samples"


## This implementation is adapted from:
## https://github.com/amazon-science/tabsyn/tree/main


## Installing Dependencies

Create conda environment

```
conda env create -f environment.yml
```


### Using your own dataset

First, create a directory for you dataset [NAME_OF_DATASET] in ./data:
```
cd data
mkdir [NAME_OF_DATASET]
```

Put the tabular data in .csv format in this directory ([NAME_OF_DATASET].csv). **The first row should be the header** indicating the name of each column, and the remaining rows are records.

Then, write a .json file ([NAME_OF_DATASET].json) recording the metadata of the tabular, covering the following information:
```
{
    "name": "[NAME_OF_DATASET]",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv"
    "test_path": null,
}
```
Put this .json file in the .Info directory.

Finally, run the following command to process the UDF dataset:
```

```

## Training Tabsyn Model first

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, magic, beijing, news
Options of [NAME_OF_BASELINE_METHODS]: smote, goggle, great, stasy, codi, tabddpm

For Tabsyn, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```


## CtrTab (control module) Training and Sampling

'''
python train_controlnet.py --dataname=[NAME_OF_DATASET] --device=0 --scale=[NOISE_SCALE] --save_name=CtrTab

python sample_controlnet.py --dataname=[NAME_OF_DATASET] --device=0 --scale=[NOISE_SCALE] --save-name=CtrTab --model-name=CtrTab
'''

The default save path is "synthetic/[NAME_OF_DATASET]/CtrTab.csv"
