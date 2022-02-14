# ElDet
我们提出了一个Anchor-free的通用椭圆目标检测器，可以根据目标形状信息更好地检测任意类别的椭圆目标，并经过少许修改即可应用于下游任务，例如人脸检测。
<img src="https://user-images.githubusercontent.com/76420705/153814852-4b6d4810-dd49-4c26-b0f5-8e1363d3f32f.png" alt="模型框架" width="810" height="340" align="bottom" />

## 检测结果
### 自建数据集
<img src="https://user-images.githubusercontent.com/76420705/153815397-849209a0-e407-4c18-b42e-e4d23ea6d412.PNG" alt="自建数据集结果" width="650" height="400" align="bottom" />

### FDDB数据集
<img src="https://user-images.githubusercontent.com/76420705/153815436-7122d9bc-31c7-4af6-9a39-37d3a19f308a.PNG" alt="FDDB数据集结果" width="550" height="400" align="bottom" />

## 环境配置
1. 基础环境使用文档中的requirements.txt；
2. 可形变卷积DCN配置。
