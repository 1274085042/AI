# 实验环境
* Linux  
* Python3  
* TensorFlow

# 数据集  
拥有5个分类，共155张图片，每张图片都做了标注，标注数据格式与voc数据集相同。数据地址如下：
https://gitee.com/ai100/quiz-w8-data   
  
1. 下载后的文件夹为`quiz-w8-data/`

数据集中的物品分类如下：
- computer
- monitor
- scuttlebutt
- water dispenser
- drawer chest


数据集中各目录如下
- images， 图片目录，数据集中所有图片都在这个目录下面。
- annotations/xmls, 针对图片中的物体，使用LabelImg工具标注产生的xml文件都在这里，每个xml文件对应一个图片文件，每个xml文件里面包含图片中多个物体的位置和种类信息。     
- labels_items.txt，数据集中的label_map文件  
  
## 准备数据  
2. 配置环境变量   
   在 .bashrc中添加PYTHONPATH环境变量  
    export PYTHONPATH=/root/object-detection/research:/root/object-detection/research/slim:$PYTHONPATH  
3. 代码中需要预先编译一些proto buffer的class，不然会出现如下错误
```
Traceback (most recent call last):
  File "Data_preprocessing.py", line 16, in <module>
    from object_detection.utils import label_map_util
  File "/path/to/object_detection/utils/label_map_util.py", line 22, in <module>
    from object_detection.protos import string_int_label_map_pb2
ImportError: cannot import name string_int_label_map_pb2
```
解决方式是直接在research目录下运行如下代码
```sh
sudo apt-get install protobuf-compiler
protoc object_detection/protos/*.proto --python_out=.
```  
4.  制作训练tfrecord和验证tfrecord  

    `python3 quiz-object-detection/create_data.py --data_dir ./quiz-w8-data/ --label_map_path ./quiz-w8-data/labels_items.txt --output_dir ./`    
5. 将`quiz-w8-data/labels_items.txt` 拷贝到`data`文件夹

**注意：在data文件夹，已经有准备好的`pet_train.record、pet_val.record、labels_items.txt`，上面的1、4、5步可以省略。2、3步不能省略**  

# 预训练模型
TensorFlow中的object_detection框架提供了一些预训练的模型以加快模型训练的速度，不同检测框架的预训练模型不同，本次实验使用mobilenet_v1特征提取器、ssd检测框架，其预训练模型可以在在model_zoo中找到:
https://github.com/tensorflow/models/blob/r1.5/research/object_detection/g3doc/detection_model_zoo.md  

6. 下载好的预训练模型在文件夹`ssd_mobilenet_v1_coco_2017_11_17`中  
7. 将`ssd_mobilenet_v1_coco_2017_11_17`中的`model.ckpt.index、model.ckpt.data-00000-of-00001、model.ckpt.meta`拷贝到`data`文件夹下  

**注意：在data文件夹，已经有准备好的`model.ckpt.index、model.ckpt.data-00000-of-00001、model.ckpt.meta`，上面的6、7步可以省略。**     

现在，data文件包含以下文件：  
- model.ckpt.data-00000-of-00001  预训练模型相关文件
- model.ckpt.index  预训练模型相关文件
- model.ckpt.meta  预训练模型相关文件
- labels_items.txt  数据集中的label_map文件
- pet_train.record  数据准备过程中，从原始数据生成的tfrecord格式的数据
- pet_val.record  数据准备过程中，从原始数据生成的tfrecord格式的数据


# 配置文件  
**8. 配置文件为`data/ssd_mobilenet_v1_pets.config`**     
* 修改配置文件中的预训练模型路径 [fine_tune_checkpoint](data/ssd_mobilenet_v1_pets.config#L158)   
* 修改 train_input_reader 中的 [input_path](data/ssd_mobilenet_v1_pets.config#L177)和 [label_map_path](data/ssd_mobilenet_v1_pets.config#L179)
* 修改 eval_input_reader 中的 [input_path](data/ssd_mobilenet_v1_pets.config#L191)和 [label_map_path](data/ssd_mobilenet_v1_pets.config#L193)  
* [num_classes](./data/ssd_mobilenet_v1_pets.config#L9) 原文件里面为37,这里的数据集为5
* [num_examples](data/ssd_mobilenet_v1_pets.config#L183) 这个是验证集中有多少数量的图片，请根据图片数量和数据准备脚本中的生成规则自行计算。

* [num_steps](data/ssd_mobilenet_v1_pets.config#L164) 这个是训练多少step
* [max_evals](data/ssd_mobilenet_v1_pets.config#L186)，这个是验证每次跑几轮，这里直接改成1即可，即每个训练验证循环只跑一次验证。
* [eval_input_reader](data/ssd_mobilenet_v1_pets.config#L189) 里面的shuffle， 这个是跟eval步骤的数据reader有关，如果不使用GPU进行训练的话，这里需要从false改成true，不然会导致错误，详细内容参阅 https://github.com/tensorflow/models/issues/1936

根据数据集图片总数all_images_count和batch_size的大小，可以计算出epoch的数量，最后输出模型的质量与epoch的数量密切相关。  
epoch=(num_step*batch_size)/all_images_count。


# 训练  
**9. `python3 research/object_detection/train.py --train_dir=./train_dir --pipeline_config_path=./data/ssd_mobilenet_v1_pets.config `** 

`--train_dir=./train_dir `:存放训练过程中的checkpoint和summary文件。  

# 验证  
**`10. python3 research/object_detection/eval.py --checkpoint_dir=./train_dir --eval_dir=./eval_dir --pipeline_config_path=./data/ssd_mobilenet_v1_pets.config`**  

![][image1]
![][image2]  

# 导出模型     
`python3 research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ./data/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix ./train_dir/model.ckpt-500 --output_directory ./exported `  

# 使用导出模型
在data文件夹下放一张测试图片test.jpg    

`python3 research/object_detection/inference.py --output_dir exported/ --dataset_dir ./data/`    
预测图片在`exported`文件夹  

![][image3]  
![][image4]

[//]: #(image)
[image1]:./example/1.png
[image2]:./example/2.png
[image3]:./data/test.jpg
[image4]:./data/output.png