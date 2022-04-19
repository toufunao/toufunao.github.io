---
layout:     post   				    # 使用的布局（不需要改）
title:      LR without 3rd Party 				# 标题 
subtitle:    #副标题
date:       2022-04-19 				# 时间
author:     Chris 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Federated Learning
    - Privacy Computing
---

## 1. Overview of LR

LR的最基本的形式是使用logistic函数

$P(y=1|x;\theta)=h_{\theta}(x)=\frac{1}{1+e^{-\theta x}}$

训练的目标函数是：

$L=-\frac{1}{n}\sum_{i=1}^ny_ilogh_{\theta}(x_i)+(1-y_i)log(1-h_{\theta}(x_i))$

梯度为：

$\frac{\partial L}{\partial \theta}=-\frac{1}{n}\sum_{i=1}^n(y_i-h_{\theta}(x_i))x_i$

对于$\theta x$来说，线性预测可以写成

$\theta x=\theta^Ax^A+\theta^Bx^B$

## 2. 训练流程

|       | Party A                                                      | Party B                                                      |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Step0 | 生成加密密钥对，将公钥发送给B                                |                                                              |
| step1 | 初始化$\theta^Ax_i^A$                                        | 初始化$\theta^Bx_i^B$                                        |
| step2 | 计算$\theta^Ax_i^A$                                          | 计算$\theta^Bx_i^B$并发送给A                                 |
| step3 | 计算$\theta x_i=\theta^Ax_i^A + \theta^Bx_i^B,\hat{y_i}=h_{\theta}(x_i),[[y-\hat{y_i}]],并且将[[y-\hat{y_i}]]发给B$ |                                                              |
| step4 | 计算$\frac{\partial L}{\partial\theta^A}$ 和loss L           | 计算$[[\frac{\partial L}{\partial \theta^B}]]$ 生成一个随机数$R_B$，并且将$$[[\frac{\partial L}{\partial \theta^B}]]+[[R_B]]$$发送给A |
| Step5 | 解密$[[\frac{\partial L}{\partial \theta^B}]]+[[R_B]]$，将$\frac{\partial L}{\partial \theta^B}+R_B$发送给B |                                                              |
| step6 | 更新$\theta^A$                                               | 更新$\theta^B$                                               |

