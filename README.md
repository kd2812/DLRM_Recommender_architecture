# DLRM_Recommender_architecture

This is a modular pytorch implementation of the DLRM paper for Click Through rate prediction.

Paper Link: [[DLRM Paper](https://arxiv.org/abs/1906.00091)]

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

