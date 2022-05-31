# 基于Resnet系列网络对FER2013数据集对人脸表情进行分类
## 文件说明
- ./configs/fer2013_config.json : 存放训练参数配置文件
- ./model : 定义网络文件夹
- ./utils : 定义训练过程中Acc计算等功能
- ./dataset.py : 网络训练前数据预处理部分
- ./main.py : 主要训练代码
- ./radam.py : 优化器定义
- ./svm.py : 使用SVM作为分类器对resnet提取的特征进行分类主要代码
