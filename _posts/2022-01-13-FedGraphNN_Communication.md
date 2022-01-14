---
layout:     post   				    # 使用的布局（不需要改）
title:      FedGraphNN通信机制详解 				# 标题 
subtitle:   以MPI方式为例 #副标题
date:       2022-01-13 				# 时间
author:     Chris 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Federated Learning
    - FL Framework
    - FedGraphNN
---

## 一.FedGraphNN

[FedGraphNN](https://github.com/FedML-AI/FedGraphNN)是[Chaoyang He](https://chaoyanghe.com/)大神在[FedML](https://github.com/FedML-AI/FedML)基础上，进一步封装专门进行图神经网络训练的联邦学习框架。论文将图神经网络的联邦学习分为了两个部分，GNN训练和联邦学习更新。GNN训练分为两个阶段，Message-passing和Readout。Message-pass顾名思义是将图节点周围的节点信息进行聚合，之后利用聚合后的信息进行节点的隐含状态的更新。Readout阶段根据图神经网络任务的不同，需要将最后一层网络的值进行计算，为下游的任务提供预测。联邦学习则是依据不同训练节点获得的神经网络参数进行全局更新，将全局更新后的参数返回给各个训练节点，随后进行下一轮的训练。目前支持的图神经网络有GCN、GAT、GraphSage、SGC以及GIN，支持的联邦学习算法有FedAvg、FedOPT和FedProx等。

FedGraphNN同时提供了多种数据集进行选择使用，并根据任务和使用领域的不同，将其划分为了3大类和7小类。同时为了模拟数据的Non.IID问题，框架提供了基于隐狄利克雷分布的取样方法，生成存在Non.IID问题的训练数据进行仿真。

同时为了保证本地模型和数据的安全，FedGraphNN提供了LightSecAgg模块。

作者认为基于此平台，不同研究者可以专注于算法本身的开发，并且可以进行有效的评估。

## 二.基于MPI的同步通信

FedGraphNN提供了集中式和分布式的训练demo，存在部分的坑和bug，但是调试过后还算可以正常跑通。

FedGraphNN的demo以及使用脚本提供的都是MPI的通信形式，但是经过源码的阅读，FedGraphNN基于的FedML平台其实也已经实现了MQTT和GRPC的通信方式，但是MPI为默认通信。因此，仅以MPI作为示例进行解释。



![](https://github.com/toufunao/pic_repo/blob/main/2022-01-13/fl.jpg?raw=true)



### 2.1. 总体架构

首先我们来看整体的架构。对于联邦学习来说，Server实际的作用是对各个Client的参数进行一个Aggregate操作，经过一定算法后再将更新后参数返回给各个Client。Client的作用就是在本地上利用自己的数据进行训练，并根据Server端传来的更新后的参数进行调整。

所以对于Server和Client来说，他们所共有的就是底层的通信模块。此处我们不细讲，下一节会展开。对于Client来说，他的一个任务就是进行模型训练，因此Client必须有一个训练模块来管理各种信息。训练需要有模型、数据和设备，因此这些信息被存储在了Trainer中。但是需要注意的是，因为Client训练完一轮之后需要等待全局更新后的参数再进行下一轮的训练，为此我们必须有一个trigger来触发Client进行下一轮的训练。在FedGraphNN中，设计者巧妙的使用了观察者模式和Python的回调机制，当receive_thread收到信息时，会去触发handler，根据不同的信息类型，回调不同的方法，例如Client的初始化和训练等。

Server端同理，但是不同的是，Server端是对参数进行Aggregate，因此不需要进行训练。同时，Server启动时便会发送init消息，通知各个Client进行初始化准备。

### 2.2. MPI同步通信

```python
def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number
```

首先，MPI在初始化时会初始化一个线程组，这些线程组内的线程可以互相通信，依据的是各个线程的id。这里的comm即是线程组的标识。

MPI通信主要是由CommunicationManager和其维护的两个线程send和receive实现。

send和receive各自维护着一个缓冲队列，send每隔0.003s轮询一次自己维持的维护队列，如果有新消息放入，就将其发送。

```python
    def run(self):
        logging.debug("Starting " + self.name + ". Process ID = " + str(self.rank))
        while True:
            try:
                if not self.q.empty():
                    msg = self.q.get()
                    dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
                    self.comm.send(msg.to_string(), dest=dest_id)
                else:
                    time.sleep(0.003)
            except Exception:
                traceback.print_exc()
```

对于收到的消息，CommunicationManager线程本身对其进行轮询，时间为每隔0.3s一次。如果有消息收到，将会通知观察者，观察者拿到消息后进行利用Python的回调机制进行信息处理。

```python
 def handle_receive_message(self):
        self.is_running = True
        while self.is_running:
            if self.q_receiver.qsize() > 0:
                msg_params = self.q_receiver.get()
                self.notify(msg_params)

            time.sleep(0.3)
        logging.info("!!!!!!handle_receive_message stopped!!!")
```

## 三.设计分析

经过源码的分析，我们很容易发现，FedGraphNN的模块化设计十分精巧，任一模块的可扩展性都极强。如果需要换GNN的模型，只需要更换model模块；如果需要使用不同的联邦学习算法，可以更换Aggregator模块；如果需要对模型的处理有别的操作，可以对handler模块中的具体方法进行扩展；如果需要使用不同的通信协议，将底层的CommunicationManager进行更换即可。这一切都体现设计者对设计模式（观察者模式，面向接口编程）已经对Python语言特性（回调机制）的深刻理解和灵活运用。（Chaoyang He大佬真的不愧是在工业界工作过那么久的人，🧎‍♂️了）

## 四.FedGraphNN不足

1.目前FedGraphNN仅仅支持同步通信，但是同步通信会带来很大的性能损耗，整体训练会被响应最慢的机器拖累。在异构设备（计算资源不一致)、网络环境不佳的情况下会尤其明显，而这些恰恰是联邦学习常用的场景。

2.目前的数据集不支持图神经网络的自监督和半监督学习。

3.在大数据集实验中提及了Non.IID对性能造成的影响，但是并未提供解决问题的方法。