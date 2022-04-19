---
layout:     post   				    # 使用的布局（不需要改）
title:      From XGBoost to SecureBoost 				# 标题 
subtitle:    #副标题
date:       2022-04-10 				# 时间
author:     Chris 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Federated Learning
    - Privacy Computing
---

## 一. XGBoost

### 1. 数学推导

树模型的预测方法   $$\hat{y_i}=\sum_{k=i}^Kf_k(X_i)$$K代表子树数， $f_k(X_i)$代表$x_i$在子树k上的取值

目标函数   $J=\sum_il(\hat{y_i},y_i)+\sum_k\Omega(f_k)$,其中，$\Omega(f_k)=\gamma*T+\frac12\lambda||w||^2$ 为正则项，用来衡量模型的复杂度，可以帮助防止过拟合。T代表子树数目，w代表叶子结点的取值。

对于第i个特征在，第t轮/棵的损失值，可以表示为 

$$J^{(t)}=\sum_{i=1}^nl(y_i,\hat{y_i}^{t-1}+f_t(x_i))+\Omega(f_t)$$

进一步可以化为

$J^{(t)}=\sum_{i=1}^nl(y_i,\hat{y_i}^{t-1}+f_t(x_i))+\gamma*T+\frac12\lambda\sum_{j=1}^Tw_j^2$

泰勒二阶展开，$g_i为一阶导数，h_i为二阶导数$

​		$=\sum_{i=1}^n{(l(y_i,\hat{y_i}^{t-1})+L'(y_i,\hat{y_i}^{t-1})f_t(x_i)+\frac12L''(y_i,\hat{y_i}^{t-1})f_t^2(x_i))}+\gamma*T+\frac12\lambda\sum_{j=1}^Tw_j^2$

​		$=\sum_{i=1}^n{(l(y_i,\hat{y_i}^{t-1})+g_if_t(x_i)+\frac12h_if_t^2(x_i))}+\gamma*T+\frac12\lambda\sum_{j=1}^Tw_j^2$

$l(y_i,\hat{y_i}^{t-1})$为常数，已在前一轮中得到

​		$=\sum_{i=1}^n{(g_if_t(x_i)+\frac12h_if_t^2(x_i))}+\gamma*T+\frac12\lambda\sum_{j=1}^Tw_j^2+C$

​		$=\sum_{i=1}^n{(g_iw_{q(x_i)}+\frac12h_iw_{q(x_i)}^2}+\gamma*T+\frac12\lambda\sum_{j=1}^Tw_j^2+C$

对每棵树的结点进行计算

​		$=\sum_{j=1}^T{[(\sum_{i\in{I_j}}g_i)w_j+\frac12(\sum_{i_\in{I_j}}h_i)w_j^2]}+\gamma*T+\frac12\lambda\sum_{j=1}^T w_j^2+C$

​		$=\sum_{j=1}^T{[(\sum_{i\in{I_j}}g_i)w_j+\frac12(\sum_{i_\in{I_j}}h_i+\lambda)w_j^2]}+\gamma*T+C$

此时为关于$w_j$的二次方程

 	$\frac{\partial J^{t}}{\partial w_j}=\sum_{j=1}^T{[(\sum_{i\in{I_j}}g_i)+(\sum_{i_\in{I_j}}h_i+\lambda)w_j]}=0$

​	 $w_j^*=-\frac{\sum_{i_{\in{I_j}}}g_i}{\sum_{i_\in{I_j}}h_i+\lambda}$

带回可得

​	$J^{(t)}=-\frac12\sum_{j=1}^T\frac{(\sum_{i\in I_j}g_i)^2}{\sum_{i\in I_j}h_i+\lambda}+\gamma T$

评价分裂的指标为information gain

$InformationGain=\frac12[\frac{(\sum_{i\in{I_L}}g_i)^2}{(\sum_{i\in{I_L}}h_i)+\lambda}+\frac{(\sum_{i\in{I_R}}g_i)^2}{(\sum_{i\in{I_R}}h_i)+\lambda}-\frac{(\sum_{i\in{I}}g_i)^2}{(\sum_{i\in{I}}h_i)^2}]-\gamma$

### 2. 贪心算法和近似算法

对于贪心算法而言，执行的是遍历操作，将每个点都带入进行计算。

近似算法则是将数据分块，根据分位点的值计算寻找最佳的分裂点。分块方式分为加权分位和普通分位。

近似算法同时也分为全局和局部，全局意味着第一次筛选出来的分位点，在下层子树进行分裂时继续使用；而局部分裂则代表每次分裂前都需要重新计算寻找分位点。

### 3. 对于稀疏值的处理

将缺失的值的属性要么全部分配到左子树，要么分配到右子树，比较两次分配后的信息增益，根据最优的信息增益进行分裂。

### 4. 系统设计

1.核外块运算

将数据分成块分别存储在不同的机器上。在计算时，为了使得计算和磁盘读入并行处理，用了一个独立的线程来进行pre-fetch。

block compression：将块以列的形式进行压缩，读取的时候需要进行解压。

block sharding: 数据被分到不同的磁盘，每个磁盘都会被分配到一个pre-fetch线程。训练线程会轮流读取各个buffer的数据进行训练。

2.列分块并行学习

Feature列的数据会以csc压缩模式存储在内存中，并且排序好。feature会有指针指向instance index

不同列的统计分位点，计算可以并行执行。

3.缓存优化

贪心算法：每个列分配一个buffer，将$G_i,H_i$存入buffer中。

近似算法设置合理的block size

## 二. SecureBoost

### 1. 角色分类及SecureBoost目标

拥有数据和数据标签的一方称为主动方(active party)，只拥有数据的一方称为被动方(passive party)。

目的是结合被动方的数据和主动方的标签训练一个模型，在保证数据不泄漏的情况下，保证模型的性能与直接将数据聚合在一起的情形下训练出的模型性能是一致的（无损）。

### 2. FL with SecureBoost

1.在进行训练前，首先需要找到被动方和主动方的common data。论文中直接使用了Privacy-preserving inter-database operations论文中的方法。这个过程也被称作求交。

2.使用FL+SecureBoost进行训练。

### 3. 关键点

1.从xgboost的推导过程中，我们可以发现分裂点的评估和叶子结点权重的计算只依赖于$g_i$和$h_i$

2.标签可以通过$g_i$和$h_i$推测出来，例如如果使用平方损失，$g_i=2*(\hat{y_i}^{t-1}-y_i)$

由1可知，client可以通过本地的数据和$g_i,h_i$计算出本地的最优分裂点；

由2可知，$g_i,h_i$是敏感数据，client可以由他们反推出label，因此需要加密。

### 4. 训练流程

1.Client根据各自feature计算分位点，并将$[G_{kv}],[H_{kv}]$发送给Server

2.Server根据分位点计算信息增益，选择最大的分位点，返回给Client。

3.Client根据Server返回的分位点进行分裂，并返回$<recordId,I_L>$

4.Server进行分裂，记录$<partyId, recordId >$

### 5. lossless

同态加密

$<m_1><m_2>=<m_1+m_2>$

$<H>=\prod_{i\in{I_L}}<h_i>=<\sum_{i\in{I_L}}h_i>$

$<G>=\prod_{i\in{I_L}}<g_i>=<\sum_{i\in{I_L}}g_i>$

