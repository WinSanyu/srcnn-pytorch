# SRCNN

参考http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html

## 训练

将91数据集或291数据集放入`datacrop/Train`

运行`datacrop/generate_train.m`生成训练集

运行`datacrop/generate_test.m`生成测试集

（SRCNN的测试集和别的SR网络不太一样）

将Test_sub_bic,Test_sub,Train_sub_bic,Train_sub移动到

`code/data`

运行`code/main_srcnn.py --cuda --gpus 0`

## 测试

运行`code/eval --model chekpoint/model_epoch_21.pth`