# DLRM_Recommender_architecture

This is a modular pytorch implementation of the DLRM paper for Click Through rate prediction.

Paper Link: [[DLRM Paper](https://arxiv.org/abs/1906.00091)]

## Run it
```
pip install -r requirements.txt
# change the directory addresses of the files 
```

## Notes from the paper:

- DLRM is a combination of the all the models that came before it.
- The dataset can be divided in to dense and sparse features.
- Let there be $n_1$ dense features and $n_2$ sparse features.
- The $n_1$ dense features are made to go through a MLP[Bottom MLP] which outputs a vector $D \in R^d$.
- The sparse features are embedded through an Embedding layer and become $e_1,e_2,...,e_{n_2} \in R^d$.
- There latent features of the sparse features can further optionally be passed through one more MLP.
- Basially both the latent feature sets get the same dimensions.
- Now, we take the second order feature interactions via dot product of each unique pair of $D,e_1,e_2,...,e_{n_2}$.

- The dot products are all scalars $d_1,d_2,...$.
- They are concatinated with the D vector, like, $[D,d_1,d_2,..]$.
- Now, this new vector is passed through a MLP[Top MLP].

## Implementation Details
- framework used: Pytorch
- Embedding: nn.Embedding
- Loss: BCEWithLogitsLoss

## Dataset Details:
- Criteo style dataset
- 13 integer type features [dense features]
- 26 categorical features [sparse features]
- Total rows in training dataset: 40,000
- Out of that : 32k for training and 8k for validation
- Total rows in test set: 10,000
- Dataset is highly imbalanced with around 3.2% positive labels

## Metrics used:
- ROC-AUC
- Log-Loss
- F1-Score
- Accuracy
- Balanced Accuracy
- PR-AUC

## Data preprocessing:
- signed log transform on numerical categories
- quantile clipping of numerical categories[1-q to q percentile are kept with deafult q=0.999]
- binary missing data flags are added
- missing values in numerical data is replaced with median value
- categorical data are equiped with MISSING and RARE id

## Deviations from the Paper:
- BatchNorm and Dropout are added to the MLPs.
- Adam with lr 1e-4 is used, while the paper uses SGD.
- Final Sigmoid not used as BCEWithLogitsLoss is used
- The dataset is far smaller than the one the paper used.

## Limitations:
- dataset is quite small with aroun 40k columns
- No proper baseline designation
- No hyperparameter search

## Results [on Test Set]:

| S.No. | Model | ROC-AUC | Log-Loss | F1 | Accuracy | Balanced-Accuracy | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 |   DLRM small |  0.664766 |  0.143449 |  0.107539 |  0.8473 |  0.570888 |  0.065088 |
| 1 |  DLRM medium |  0.668815 |  0.143222 |  0.105263 |  0.9116 |  0.546520 |  0.062652 |
| 2 |   DLRM large |  0.670971 |  0.142037 |  0.110286 |  0.9161 |  0.548848 |  0.067479 |

## Learnings and Findings:
- All the models have a ROC-AUC of around ~0.66 to 0.67 which shows that they all performed almost the same. 
- The largest model has the highest F1 score showing that out of the three that is the best choice in regards to precision and recall. 
- Accuracy is not much informative in this case as the dataset is quite imbalanced. 
- Even PR_AUC is within the same range for all three models.
- The dataset being quite small is a huge bottleneck for these models which were originally trained in the paper on quite larger datasets.
- From the loss curve in the notebook, no sign of overfitting is evident.


## Future Improvements:
- addition of a baseline with a regular MLP or XGboost.
- hyperparameter search 
- use of a bigger dataset
- running k-fold cv