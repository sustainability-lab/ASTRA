# Active Learning Tools

## Introduction

Assume you are working in a car insurance compney. You compney is impressed by the recent advances in machine learning and wants you to propose an automatic solution to estimate the damage of a car from a picture. You can train a model to estimate this, but first you need labeled data. You can guess that labeling a car damage is not an easy task. What if you can pick the most informative images to label in a way that your model becomes more accurate with less labeled data? This is the goal of active learning. It helps you to pick the most informative data to label and train your model. But how does the model know which data is informative? That's where different active learning strategies and acquisions come to help.

## Strategies

What we call a strategy in this toolkit is, a specific way of picking the most informative data to label. For example, you may train N models on the data and pick the data that the models disagree on (`EnsembleStrategy`) or you may pick the data point which is farthest from the current training data in the embedding space (`DiversityStrategy`). You can find the implemented strategies in the `strategies` folder.

## Acquisitions

Once we fix a strategy, we can pick the most informative data points with different acquisions. For example, you may want to pick data points with highest predictive entropy (`Entropy`). You can find the implemented acquisitions in the `acquisitions` folder.

## Why have we implemented strategies and acquisitions separately?
* We want to standardize the interface of strategies while keeping the flexibility of acquisitions. For example, you may want to use a different acquisitions for a strategy. For example, `Entropy` just takes `logits` of pool data as input while `Furthest` takes `embeddings` of training and pool data as input. We can see that these inputs are vastly different from each other.
* We can standardize some processes such as, our model should always be in `train` mode while using `MCStrategy` while it should be in `eval` mode while using `EnsembleStrategy`. In short, we can make minimize the errors that we may make while using different strategies while keeping the acquisition code clean.
* To reduce redundancy. For example, `EnsembleStrategy` and `MCStrategy` both can use `Entropy` as acquisition. If we implement `Entropy` in `EnsembleStrategy`, we have to implement it again in `MCStrategy`.