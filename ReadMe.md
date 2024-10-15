
## 代码介绍

   
- 代码旨在完成对于学生作文图片，识别出错字、别字、并根据上下文进行错字的纠正的任务，具体见BLIP文件夹。
- 采用模型: blip-image-captioning-base
- 参数微调方法： Lora    

```
python

# 模型训练 见脚本： train_blip.ipynb

# 虽然训练完成但有问题：由于BLIP不支持很多中文字符的tokenizer，代码仍在修正中。
```

## 背景介绍

   该任务来自于 NLPCC2024 会议中 Task1：Task 1 - Visual Chinese Character Checking（见下说明），Task1的数据集的参考论文为：Towards Real-World Writing Assistance: A Chinese Character Checking Benchmark with Faked and Misspelled Characters。 该任务主要目标是根据学生作文图片，识别出错字、别字、并根据上下文进行错字的纠正。
   
<big>*Task说明*</big>

In the real world where handwriting occupies the vast majority, characters that humans get wrong include faked characters
(i.e., untrue characters created due to writing errors) and misspelled characters (i.e., true characters used incorrectly due to spelling errors), 
as illustrated in Figure 1. Visual Chinese Character Checking task aims at detecting and correcting wrong characters in the given text on an image in real-world scenarios. 
This task focuses on correcting characters based on a human-annotated Visual Chinese Character Checking dataset with faked and misspelled Chinese characters.



![task1](task1_1.PNG)


  以上参考论文中实现了2种基准方法，这2类基准方法均采用Detection+Correction结构处理处理预测数据。方法1利用文本识别 + 文本生成的方法，而方法2利用文本分割+多模态模型+文本生成的方法。方法1及方法2的效果都不是很理想。

<big>*参考论文方法*</big>

![method](methods_1.PNG)


<big>*参考论文效果*</big>

![effect](effect_1.PNG)
