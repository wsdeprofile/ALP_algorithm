# 用户对齐算法整合 #

## 文件结构 ##

* algs/ &nbsp; &nbsp; &nbsp; &nbsp;           算法的具体实现

* data/ &nbsp; &nbsp; &nbsp; &nbsp;           用于存放数据文件

* metrics/ &nbsp; &nbsp;         算法评价方法的具体实现

* model/  &nbsp; &nbsp;         用于存放保存的模型文件

* results/ &nbsp; &nbsp;        可以用于保存输出的结果

* utils/  &nbsp; &nbsp; &nbsp; &nbsp;         工具方法的具体实现

* run.py  &nbsp; &nbsp; &nbsp;         模型训练及测试的脚本

## 运行方法 ##

* 执行 python run.py [算法] 即可自动执行整个流程，包括数据预处理，模型训练，测试评价。
  >示例：运行MNA算法:&emsp;  
  python run.py mna

* 各种参数的调整可以通过在算法名称之后添加 -参数名 参数 来完成
  >示例：运行MNA算法,并指定模型存放路径为 model/MNA.pkl :&emsp;  
  python run.py mna -model_path model/MNA.pkl

* 评价模型的性能采用： 
  * 分类任务指标  
    * F1
    * Accuracy
    * AUC
  * 排序任务指标
    * MRR
    * Precision@1
    * Precision@5

## 进展情况 ##

目前实现算法包括：
* MNA算法 &nbsp; python run.py -mna
* STUL算法 &nbsp; python run.py -stul
* HYDRA算法 &nbsp; python run.py -hydra


## 其他说明 ##

1. 目前的实现暂未使用网络结构特征，后续会进行补充

## 参考文献 ##

1.MNA:  [Kong, Xiangnan, Jiawei Zhang, and Philip S. Yu. "Inferring anchor links across multiple heterogeneous social networks." Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.](http://web.cs.wpi.edu/~xkong/publications/papers/cikm13.pdf)  
2.STUL:  [Chen, Wei, et al. "Exploiting spatio-temporal user behaviors for user linkage." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.](https://dl.acm.org/doi/abs/10.1145/3132847.3132898)  
3.HYDRA:  [Liu, Siyuan, et al. "Hydra: Large-scale social identity linkage via heterogeneous behavior modeling." Proceedings of the 2014 ACM SIGMOD international conference on Management of data. 2014.](https://dl.acm.org/doi/abs/10.1145/2588555.2588559)  
