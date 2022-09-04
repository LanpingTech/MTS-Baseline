import numpy as np
from sklearn.metrics import pairwise_distances

def hasFalse(visited):
    for i in range(visited.size):
        if not visited[i]:
            return True
    return False


# 计算 [n, m] 区间内的 count 的平均值
def average_n_m(data, n, m):
    sum = 0
   
    data=data.reshape(-1)
    for i in range(m - n):
        sum += data[i + n]
    return sum / (m - n)


# 一次移动平均
def moving_average_std(data, windows):
    # 第一个参数是输入数据，第二个参数是 windows跨度的取值
    
    length=data.size(2)
    if windows > length:
        return "windows 的取值大于时间步长"
    # 定义 Moving_average 来记录移动平均值
    Moving_average = np.empty(length - windows)
    for i in range(length - windows):
        m = average_n_m(data, i, i + windows)
        Moving_average[i] = m
    # print(Moving_average.shape)
    # print(Moving_average)
    return Moving_average


# 算法2 基于相似度的正负对比抽样
def SIM(train, train_size, x_ref, s_ref, s, samples):
    _, input_dim, time_steps = train.shape
    # print(x_ref.shape)
    # print(s_ref)
    x_ref = x_ref.reshape(1,-1)

    # 从train中找到所有时间步长大于s_ref的时间序列的索引，存到index中，再从中随机取一个索引对应的序列
    for i in range(train_size):
        index = np.empty(train_size, dtype=int)
        
        _,time_steps_i=train[i].shape
        
        if time_steps_i > s_ref:
            for j in range(train_size):
                index[j] = i  # 存到index中

    z_pos = [] 
    z_neg = [] 

    for k in range(samples):

        # 从index中随机选出il
        il = index[np.random.randint(0, index.size)]
        _,s_il=train[il].shape # 当前序列的时间步长
        
        Sil = []
        
        # 从当前序列中依次取长度为s_ref的序列并将序列维度打平跟x_ref（也是打平后的）求欧几里得距离
        for j in range(s_il - s_ref):
            Sil.append(train[il:il + 1, :, j:j + s_ref-1].reshape(-1))
        #print(Sil.shape)
        #print(x_ref)
         
        dist=pairwise_distances(np.array(Sil), x_ref)
       
        # 距离最近/远的Sil的片段的索引起点j
        index1 = dist.argmin()
        index2 = dist.argmax()
        
        # 距离x_ref最近和最远的数据以及起点索引（步长的起点）
        start1 = np.random.randint(index1, s_ref - s + index1)
        start2 = np.random.randint(index2, s_ref - s + index2)
        # 随机选取长度为s的
        z_pos.append(train[il:il + 1, :, start1:start1 + s])
        z_neg.append(train[il:il + 1, :, start2:start2 + s])

    return z_pos, z_neg


class TripletSelection:
    def __init__(self, windows, theta):
        super(TripletSelection, self).__init__()
        self.windows = windows
        self.theta = theta

    def __call__(self, batch, train, samples):
        batch = batch[0]
        batch_size, input_dim, time_steps = batch.shape
        train_size,_,_ = train.shape
        length = time_steps  # 时间步长

        # 调用算法1 anchor_selection获得anchor list X[x_start,x_end]
        X_list = np.empty((batch_size,2),dtype=int)
        for j in range(batch_size):
            X_list[j] = self.anchor_selection(j, input_dim, length, batch)
        #print(X_list)
        x_ref = []
        pos_samples = []
        neg_samples = []
        for i in range(batch_size):
            # 随即均匀的选取长度s以及锚的长度s_ref

            x_start = X_list[i][0]
            x_end = X_list[i][1]
            # 从anchor list X[x_start,x_end]获取x_ref
            x_ref_i = batch[i:i+1, :, x_start:x_end]
            #print(x_ref_i)
            s_ref=x_end-x_start+1
            
            s = np.random.randint(1, s_ref)
          
            x_ref.append(x_ref_i)
           
             
            # 调用算法2 SIM(D，x ref, s ref)得到当前batch中每一条数据对应的k个正负样本： x_pos, x_neg
            x_pos, x_neg = SIM(train, train_size, x_ref_i, s_ref, s, samples)
            pos_samples.append(x_pos)
            neg_samples.append(x_neg)
        
        return pos_samples, neg_samples, x_ref

    # 算法1：基于交叉序列方差的对比锚点选择,需要得到锚列表中有train_size个区间
    def anchor_selection(self, train_index, num, length, batch):
        #print(train_index)
        std = np.empty(length-self.windows)
        M = np.empty((batch.size(0), num, length-self.windows))

        # 计算每一个时间序列的移动平均值,对于多变量数据moving_average_std:
        for n in range(num):
            #print(batch[train_index:train_index+1, n:n+1, :].shape)
            M[train_index, n, :] = moving_average_std(batch[train_index:train_index+1, n:n+1, :], self.windows)
        #print(M)

      
        # 在步长维度求每一个时间步长处的标准差,并在变量维度聚合标准差
        S = np.empty(length-self.windows, dtype=float)
        
        for m in range(length-self.windows):
            for n in range(num):
                #print(M[train_index:train_index+1, n:n+1, m:m+1])
                S[m] += M[train_index:train_index+1, n:n+1, m:m+1]  # 在变量维度聚合标准差
        
        #std[m] = (S[m] - np.mean(S[m])) / np.std(S[m])  # 在步长维度归一化方差
        std=S

        # 定义一个长度为length-self.windows初始化全False的bool数组
        visited = np.full(length-self.windows, False, dtype=bool)  # 全 False
        # 根据局部方差降序排列得到索引值存到p中
        p = np.argsort(std)[::-1]
         
        i = 0
        index = 0
        x_ref = np.empty((length, 2), dtype=int)
        while hasFalse(visited):
            # 如果p[i]位置已经被访问则i++,继续找下一个未被访问的方差最大的时间步
            while visited[p[i]]:
                i = i + 1
            # 以p[i]为中心确定起点和终点
            x_ref_start = p[i] - self.windows
            if x_ref_start<0:
                x_ref_start=0
            x_ref_end = p[i] + self.windows
            if x_ref_end>length-self.windows-1:
                x_ref_end=length-self.windows-1
             
            # 如果锚点的相邻锚点的方差大于theta，将继续在两个方向上扩展
            # 如果锚点的相邻锚点的visited值为True,将继续在两个方向上扩展
            neig_left=p[i]-1
            while x_ref_start>0 and std[neig_left] > self.theta:
                x_ref_start = x_ref_start-1
                neig_left = neig_left-1
            while x_ref_start>0 and visited[neig_left]:
                x_ref_start = x_ref_start-1
                neig_left = neig_left-1
                
            neig_right=p[i]+1
            while x_ref_end<length-self.windows-1 and std[neig_right] > self.theta:
                x_ref_end = x_ref_end+1
                neig_right = neig_right+1
            while x_ref_end<length-self.windows-1 and visited[neig_right]:
                x_ref_end = x_ref_end+1
                neig_right = neig_right+1
 
            # 不再扩展时，设置所有在x_ref_start,x_ref_end区间内的visited数组值为True，已访问
            #print(x_ref_end)
            
            for j in range(x_ref_start, x_ref_end+1):
                visited[j] = True

            # 把[x_ref_start:x_ref_end]存到索引对应的锚列表
            #print(x_ref_end)
            x_ref[index, :] = np.array([x_ref_start, x_ref_end])
            index = index + 1
            
        #print(x_ref[0, :])
        return x_ref[0, :]
