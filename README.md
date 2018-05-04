# Neural Network for Distribution prediction

This is the repository accompanying the paper _Neural Networks for Predicting Algorithm Runtime Distributions_.

```
@proceedings{eggensperger-ijcai18,
  author = {K. Eggensperger and M. Lindauer and H. Hoos},
  title = {Neural Networks for Predicting Algorithm Runtime Distributions},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI'18)},
  year      = {2018}
}
```

It includes scripts and notebooks for running the experiments shown in the paper.
This code has been written and tested with *Python 3.5*; all dependencies are listed in _requirements.txt_

# Data

The data used to train the networks can be found [here](http://www.ml4aad.org/wp-content/uploads/2018/04/DistNetData.zip).
After downloading, please put the content in `./data/`.

# How to train and evaluate models

1) Create predictions using `eval_model.py`

This script trains different models (DistNet, multi-output RFs, independent RFs) on different distribution families
(inverse Gaussian, Lognormal, Exponential) using crossvalidation. Running the script will train either a DistNet or
both forest-based models on one fold for a given distribution type and scenario. The predictions will be stored using pickle. 

Also see: ```python eval_model.py```

For example:
```
python eval_model.py --model lognormal_distfit.floc --scenario clasp_factoring --fold 0 --seed 100 --save ./TEST_100 --num_train_samples 100
python eval_model.py --model lognormal_nn.floc --scenario clasp_factoring --fold 0 --seed 100 --save ./TEST_100 --num_train_samples 100
```

**NOTE:** To perform the full crossvalidation and the reproduce the results from the paper you need to train each model on folds `[0, 1, ..., 9]` using seeds `[100, 200, ..., 1000]` 
for each distribution and number of training samples `[1, 2, 4, 8, 16, 32, 64, 100]`.

2) Analyse results using one of the two jupyter-notebooks provided in `/notebooks/`

**CreateTable_evalModel-MultiSeed**

Creates a table with average NLLHs for each scenario and model

**PlotSubsets_evalModel-MultiSeed**

Creates plots that show average NLLHs compared to the number of observations per instance used for training the model.

# Further notes

On how to train DistNets and preprocess data, please have a look at the script `eval_model.py`. 
Also, please have a look at the other notebooks which provide further options to visualize and analyze runtime data.