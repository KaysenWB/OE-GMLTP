# OE-GMLTP
Ship trajectory prediction in long-term multi-ship scenarios. Implemented by Multi-Graph Convolutional Network (MGSC) and probabilistic sparse self-attention mechanism (Probsparse attention).
This code is for the paper "Graph-driven Multi-vessel Long-term Trajectories Prediction for Route Planning under Complex Waters".

The input is trajectory data with the shape [length, batch, feature], and a corresponding adjacency matrix with the shape [length, nodes, nodes], which can be referred to the following code for data processing: https://github.com/KaysenWB/AIS-Process. Please note that setting the parameter MGSC to "True"


![Figure01](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure01.jpg?raw=true)
General overview figure of the paper. It includes data processing, trajectory prediction, and route planning.

![Figure02](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure02.jpg?raw=true)

Our proposed algorithmic flow for GMLTP on time series prediction. The key point is to perform Q-matrix sparsification before self-attention. We measure the effectiveness of each qi in the attention computation based on the gap (KL scatter) between the data distribution of the dot product pairs of qi over the K matrix and the uniform distribution. The more effective qi vectors are then filtered proportionally to the length of the trajectory input to form a sparse matrix of Q. The sparse computation is ultimately used to reduce the complexity of self-attention and achieve longer-term prediction.

![Figure03](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure03.jpg?raw=true)
Presentation of prediction results, which are based on one month's AIS data for Victoria, Hong Kong.

# Acknowledgement
The algorithm in this work references a lot of the following work: https://github.com/zhouhaoyi/Informer2020.git.
Their outstanding contributions are greatly appreciated.
