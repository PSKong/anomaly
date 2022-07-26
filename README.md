ConGNN
============


This is the source code of paper: [Controlled Graph Neural Networks with Denoising Diffusion for Network Anomaly Detection].

ConGNN aims at being generalized to new environments by learning causal relations for anomaly detection


## Requirements
python==3.7.3

pytorch>=1.4
DGL>=0.4.2
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1

## Usage
```python main.py```

## Folder descriptions
*datesets:* This is used to extract the core subgraph of a given node based on causal explanations for GNNs.

*networks:* This is used to train the representation generator which takes random noises as input and outputs synthetic representations for producing counterfactual subgraphs.

*train:* This is used to produce counterfactual subgraphs based on extracted representations.

*train_log:* This is used to conduct anomaly detection with varying environments based on a few labeled anomalies.

## Motivation for ConGNN
Anomaly detection aims to identify the anomalies which are considered as data objects deviating dramatically from the majority. It has a wide range of applications from detecting network attacks in cybersecurity, and inspecting fraudulent transactions in finance, to investigating diseases in healthcare. Recently, remarkable improvement has been achieved by taking advantage of different deep learning techniques, such as Autoencoders, GANs and few-show learning. Despite the improvements achieved, existing methods still struggle in generalization beyond training data distribution, where a well-trained model might suffer from performance degradation when applying to newly observed nodes with different environments.

CFAD tries to learn causal relations to train a robust model for anomaly detection on attributed networks. Based on the concept of Structural Causal Models.



## ConGNN Overview
Based on the concept of Structural Causal Model, CFAD interpret the generation of a node representation by graph neural networks (GNNs) as a causal process. The research of causal explanations for GNNs, and the preliminary analyses show that the neighbors of a node can be split into important ones which can influence the semantics of this node, and unimportant ones which can, to an extent, impact detection probability but hardly change the node's semantics. Hence, we decompose this process into two mechanisms: one generates the core feature and another generates the environment feature. According to learning processes of GNNs that node representations are produced by aggregating neighboring nodes in the graph, we assume that the core feature is extracted based on itself and its important neighborhood, which comprises the core subgraph, and the environment feature is extracted based on other unimportant neighborhood, which comprises the environment subgraph. We steer a generative model to manufacture synthetic environment subgraphs and then generate counterfactual subgraphs, i.e., unseen combinations of core subgraphs and synthetic environment subgraphs. Subsequently, we train an anomaly detection model based on the generated counterfactual subgraphs, such that the model can learn transferable and causal relations across different environments.
