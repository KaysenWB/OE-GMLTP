# OE-GMLTP
Ship trajectory prediction in long-term multi-ship scenarios. Implemented by Multi-Graph Convolutional Network (MGSC) and probabilistic sparse self-attention mechanism (Probsparse attention).

The implementation code for the paper "Graph-driven Multi-vessel Long-term Trajectories Prediction for Route Planning under Complex Waters".

![Figure01](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure01.jpg?raw=true)
General overview Figure of the paper. It includes data processing, trajectory prediction, and route planning.

![Figure02](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure02.jpg?raw=true)

Our proposed algorithmic flow for GMLTP on time series prediction. The key point is to perform Q-matrix sparsification before self-attention. We measure the effectiveness of each qi in the attention computation based on the gap (KL scatter) between the data distribution of the dot product pairs of qi over the K matrix and the uniform distribution. The more effective qi vectors are then filtered proportionally to the length of the trajectory input to form a sparse matrix of Q. The sparse computation is ultimately used to reduce the complexity of self-attention and achieve longer-term prediction.

![Figure03](https://github.com/KaysenWB/OE-GMLTP/blob/main/Figure03.jpg?raw=true)
Presentation of prediction results, which are based on one month's AIS data for Victoria, Hong Kong.
