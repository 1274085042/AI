# 用slim框架内预训练的inception_V3实现书法汉字的识别
## 训练日志
![][image1]

## 评估
![][image2]

## tensorboard
### SCALARS
![][image3]    

![][image4]  

![][image5]

### IMAGES  
![][image6]
### GRAPHS       
![][image7]
### DISTRIBUTIONS  
![][image8]
### HISTOGRAMS    
![][image9]  
  
## 模型训练流程
1. 准备数据集  
Tinymind汉字识别数据集（https://www.tinymind.cn/competitions/41）       
下载训练集，将解压后的数据放到`./train/ `    

2. 数据预处理   
   将图像数据转换为tfrecord格式的二进制数据，提高数据存储和读取的效率。      
   进入quiz-word-recog文件夹，执行：  

`python download_and_convert_data.py --dataset_dir=../train --dataset_name=quiz
`  

   数据预处理的一个重要环节是对数据进行增广，数据增广的方式分为有监督、半监督和无监督。此次实验采取的增广方法是有监督增广，包括几何变换和颜色变换等。数据增广也是防止模型过拟合的方法之一。另外此实验是对汉字进行识别，不能对图像进行随机裁剪。    

3. 预训练权重  
   下载 [inception_v3的预训练权重](https://github.com/tensorflow/models/tree/master/research/slim)。  
4. 基于tensorflow的slim框架进行训练      
 模型训练前需要设置超参数，如学习率，batch_size，优化器的选择等，另外进行fine-tune时需注意预训练模型的num_class是1000，而本次实验的类别是100，所以需舍去预训练模型含有num_class的权重参数，训练适合本实验的模型。  
 command：
   `python quiz-word-recog/train_image_classifier.py --dataset_name=quiz --dataset_dir=./train --checkpoint_path=./inception_v3.ckpt --model_name=inception_v3 --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/Auxlogits --train_dir=./logs --learning_rate=0.001 --optimizer=rmsprop --batch_size=16`      
5. 观察模型的训练过程，如准确率和loss的变化。待模型收敛后，对模型进行验证，数据量较大时，选择部分数据进行验证。若数据集较小可选择全部数据进行验证，查看模型的性能一般有top5 recall、准确率等。    
command：
   `python quiz-word-recog/eval_image_classifier.py --dataset_name=quiz --dataset_dir=./train --dataset_split_name=validation --checkpoint_path=./logs --model_name=inception_v3  --eval_dir=./eval  --batch_size=16 --max_num_batches=128`    

   查看TensorBoard命令：  
`tensorboard --logdir=./logs`  

6. 导出模型


[//]: #(image)
[image1]:./example/1.png
[image2]:./example/2.png
[image3]:./example/3.png
[image4]:./example/4.png
[image5]:./example/5.png
[image6]:./example/6.png
[image7]:./example/7.png
[image8]:./example/8.png
[image9]:./example/9.png