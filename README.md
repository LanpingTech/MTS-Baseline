# 多变量时间序列表示学习-对比算法

## 1. TST: A transformer-based framework for multivariate time series representation learning (SIGKDD'21)

论文简介：

TST 首次将Transformer Encoder结构应用于多变量时间序列的无监督表示学习。在TST的无监督预训练阶段，通过将部分输入数据设置为 0，并要求Transformer Encoder作为编码器，一个线性层作为解码器，MSE作为损失函数来预测被遮挡的数据，以此学习多变量时间序列的表示。经过预训练阶段后的Transformer Encoder将作为特征提取器，计算多变量时间序列的表示，来完成下游分类或回归任务。遗憾的是，TST 对超长序列和较多变量的时间序列的表示学习效果不佳。原因在于，Transformer Encoder 自身计算复杂度高的局限性。

源码链接：https://github.com/gzerveas/mvts_transformer

本项目地址：[TST](TST/)

## 2. TNC: Unsupervised representation learning for time series with temporal neighborhood coding (ICLR'21)

论文简介：

TNC 提出了一种用于学习复杂非平稳多变量时间序列表示的自监督框架。TNC 引入了平稳属性的时间邻域的概念，作为相似窗口的分布依据，对三元组损失函数中正负样本策略进行了改进。由于 TNC 的模型结构较为简单，多层 RNN + 一层 Linear 作为编码器，因此 TNC 对超长序列和较多变量的时间序列的特征提取能力较差。

源码链接：https://github.com/sanatonek/TNC_representation_learning

本项目地址：[TNC](TNC/)

## 3. TSTCC: Time-Series Representation Learning via Temporal and Contextual Contrasting (IJCAI'21)

论文简介：

TS-TCC 提出了一种基于时序和上下文的无监督对比学习框架。通过对多变量时间序列采用jitter + scale和 permutation + jitter两种不同的数据增强方式，TS-TCC分别生成了弱增强数据和强增强数据。两种不同增强数据分别进入编码器后，进行时序上和上下文上的对比学习，以此提高多变量时间序列表示的鲁棒性。在时序上的对比学习策略中，Transformer Encoder作为时序特征提取器计算复杂度较高，对超长序列处理效果不佳。TS-TCC同样忽略了多变量时间序列中变量间关系的特征提取，因此对变量数较多的时间序列的特征提取效果不佳。

代码链接：https://github.com/emadeldeen24/TS-TCC

本项目地址：[TSTCC](TSTCC/)

## 4. ReTriM: Reconstructive Triplet Loss for Learning Reduced Embeddings for Multi-Variate Time Series (ICDMW'21)

论文简介：

ReTriM 以 LeNet-5 作为编码器用于提取多变量时间序列特征，倒序 LeNet-5 作为解码器解码重构序列信息。ReTriM 采用 MAE 重构损失和 Triplet Loss 三元组损失函数想结合的方式来训练模型，其中正负样本分别选取同一batch内欧式距离最近和最远的样本。与现有算法不同的是，ReTriM 的三元组损失函数采用欧氏距离作为相似性度量方式，而不是向量内积。这样的做法会因表示的维度过大而失效。

代码链接：无

本项目地址：[ReTriM](ReTriM/)

## 5. TS2Vec: Towards universal representation of time series (AAAI'22)

论文简介：

TS2Vec 提出了一种多尺度的多变量时间序列表示对比学习框架。得利于TCN模型结构中的扩张卷积模块，TS2Vec可以提取到不同时序特征提取尺度下的序列特征信息。在每个不同尺度上，TS2Vec采用了 Temporal Contrastive Loss 与 Instance-wise Contrastive Loss相结合的方式进行时间序列的表示对比学习。 每个尺度的损失值相加作为最终的损失值。TS2Vec的实验结果表明，TS2Vec在多个时间序列数据集上的表现均优于现有算法。

代码链接：https://github.com/yuezhihan/ts2vec

本项目地址：[TS2Vec](TS2Vec/)

## 6. TSCTS: Time Series Representation Learning with Contrastive Triplet Selection (CODS-COMAD'22)

论文简介：

TSCTS 提出了一种新型的多变量时间序列表示对比学习策略。TSCTS 同样采用TCN作为序列特征提取器，基于向量内积的三元组对比损失作为损失函数。与现有算法不同的是，TSCTS 的正负样本采样策略上加入了Cross Series Variance (交叉序列方差)作为选择的依据。方差交小的样本作为正样本，反之作为负样本。

代码链接：无

本项目地址：[TSCTS](TSCTS/)

# 数据集

The basic information of data set as follows.

|     Datasets     | Dataset Full name | Train Size | Test Size | Series Length | Dimensions | Classes |
| - | :- | -: | -: | -: |-: | -: |
|1. AWR|ArticularyWordRecognition|275|300|144|9|25|
|2. AF|AtrialFibrillation|15|15|640|2|3|
|3. BM|BasicMotions|40|40|100|6|4|
|4. CT|CharacterTrajectories|1422|1436|182|3|20|
|5. CR|Cricket|108|72|1197|6|12|
|6. DDG|DuckDuckGeese|50|50|270|1345|5|
|7. EW|EigenWorms|128|131|17984|6|5|
|8. EP|Epilepsy|137|138|206|3|4|
|9. EC|EthanolConcentration|261|263|1751|3|4|
|10. ER|ERing|30|270|65|4|6|
|11. FD|FaceDetection|5890|3524|62|144|2|
|12. FM|FingerMovements|316|100|50|28|2|
|13. HMD|HandMovementDirection|160|74|400|10|4|
|14. HW|Handwriting|150|850|152|3|26|
|15. HB|Heartbeat|204|205|405|61|2|
|16. IW|InsectWingbeat|25000|25000|30|200|10|
|17. JV|JapaneseVowels|270|370|29|12|9|
|18. LB|Libras|180|180|45|2|15|
|19. LSST|LSST|2459|2466|36|6|14|
|20. MI|MotorImagery|278|100|3000|64|2|
|21. NATOPS|NATOPS|180|180|51|24|6|
|22. PD|PenDigits|7494|3498|8|2|10|
|23. PEMS-SF|PEMS-SF|267|173|144|963|7|
|24. PH|Phoneme|3315|3353|217|11|39|
|25. RS|RacketSports|151|152|30|6|4|
|26. SRSCP1|SelfRegulationSCP1|268|293|896|6|2|
|27. SRSCP2|SelfRegulationSCP2|200|180|1152|7|2|
|28. SAD|SpokenArabicDigits|6599|2199|93|13|10|
|29. SWJ|StandWalkJump|12|15|2500|4|3|
|30. UWGL|UWaveGestureLibrary|120|320|315|3|8|



