# 多变量时间序列表示学习-对比算法

## 1-TST. A transformer-based framework for multivariate time series representation learning (SIGKDD'21)

源码链接：https://github.com/gzerveas/mvts_transformer

**未完成**

## 2-TNC. Unsupervised representation learning for time series with temporal neighborhood coding (ICLR'21)

源码链接：https://github.com/sanatonek/TNC_representation_learning

**Config 未完成**

## 3-TSTCC. Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC IJCAI'21)

代码链接：https://github.com/emadeldeen24/TS-TCC

**已完成**

## 4. ReTriM: Reconstructive Triplet Loss for Learning Reduced Embeddings for Multi-Variate Time Series (ReTriM ICDMW'21)

代码链接：无

**已完成**

**虽然下游任务是分类，但是模型在训练过程中是无监督的。对比侧重点：模型结构 + 正负样本采样策略 + 损失函数**

## 5. TS2Vec: Towards universal representation of time series (TS2Vec AAAI'22)

本文介绍了 TS2Vec，这是一个用于在任意语义级别学习时间序列表示的通用框架。与现有方法不同，TS2Vec 以分层方式在增强的上下文视图上执行对比学习，从而为每个时间戳提供强大的上下文表示。此外，为了获得时间序列中任意子序列的表示，我们可以对相应时间戳的表示进行简单的聚合。我们对时间序列分类任务进行了广泛的实验，以评估时间序列表示的质量。因此，TS2Vec 在 125 个 UCR 数据集和 29 个 UEA 数据集上实现了对无监督时间序列表示的现有 SOTA 的显着改进。学习到的时间戳级表示在时间序列预测和异常检测任务中也取得了优异的成绩。在学习表示之上训练的线性回归优于之前的时间序列预测 SOTA。此外，我们提出了一种将学习表示应用于无监督异常检测的简单方法，该方法在文献中建立了 SOTA 结果。

代码链接：https://github.com/yuezhihan/ts2vec

**TCN改进 + 对比学习。对比侧重点：模型结构 + 数据增强采样策略 + 损失函数**

## 6. Time Series Representation Learning with Contrastive Triplet Selection (TS-CTS CODS-COMAD'22)

表征学习凭借其在视觉和文本模式中的吸引力和优势，已经扩展到数字时间序列。最近的工作提出了一种基于随机三元组采样的三元组损失公式，以导出时间序列的固定长度嵌入。然而，与图像和文本不同的是，统计和距离测量可以很容易地从数值时间序列数据中计算出来，以定量对比差异或基于相似性的聚类。本文通过对比识别方法研究三元组挖掘问题，在原始信号空间以及嵌入向量空间中选择锚点、正样本和负样本。然后通过因果时间神经网络学习选定的三元组，以最小化锚点到正例的距离并最大化其到负例的距离。实验结果以及消融研究比较了这些方法在分类准确度和方差方面的测量结果，证明了随机三元组选择的显着改进。我们还调查并报告了采样避免标签污染时的性能改进，展示了所提出算法的优势。

代码链接：已实现

**TCN + 对比学习。对比侧重点：数据增强采样策略 + 损失函数**

## 7. Unsupervised Time-Series Representation Learning with Iterative Bilinear Temporal-Spectral Fusion (BTSF ICML'22)

无监督/自监督时间序列表示学习是一个具有挑战性的问题，因为它具有复杂的动力学和稀疏的注释。现有的工作主要采用对比学习的框架和基于时间的增强技术对正负样本进行对比训练。然而，他们大多使用从时间切片派生的分段级增强，这可能会由于全局上下文的丢失而导致采样偏差和错误的优化以及假阴性。此外，他们都没有注意将光谱信息纳入特征表示。在本文中，我们提出了一个统一的框架，即双线性时谱融合（BTSF）。具体来说，我们首先在整个时间序列上使用带有简单 dropout 的实例级增强，以最大限度地捕获长期依赖关系。我们设计了一种新颖的迭代双线性时谱融合来显式编码丰富的时频对的亲和性，并使用频谱到时间（S2T）和时间到频谱以融合和压缩的方式迭代地细化表示。 T2S) 聚合模块。我们首先对时间序列的三大任务进行下游评估，包括分类、预测和异常检测。实验结果表明，我们的 BTSF 始终显着优于最先进的方法。

**TCN改进 + 对比学习。对比侧重点：模型结构 + 数据增强采样策略 + 损失函数**



