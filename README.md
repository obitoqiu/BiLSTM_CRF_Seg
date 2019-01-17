# bi-lstm_crf_seg
## Use lstm and crf  in keras  to achieve Chinese word segement
### 环境
* **Environments：Python3.5**</br>
* **Libs：Keras2.2.4,gensim3.6.0,keras-contrib2.0.8**
### 简介
**`preprocess.py`:      文件预处理（将分好词的训练集转换为三标签（B:词头 I:词中 S:单字词）序列标注的形式）</br>
</br>
`process.py`:         将文本序列化为模型输入、输出标准格式</br>
</br>
`LSTM_model.py`:      目前基本框架 embedding + bi_lstm + lstm + crf          </br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;词向量维度50，最长切分序列50，units=128，epoch=5...</br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;可调用model.summary()查看具体参数</br>
</br>
`train.py`:           训练模型</br>
</br>
`predict.py`:         生成预测文件</br>
</br>
`test.py`:            使用gensim生成简单的词向量</br>
</br>
`score.perl`:         sighan2004分词竞赛评测脚本**</br>
### 运行顺序
**`test.py`（也可选择其他本地词向量）</br>
`preprocess.py`—>`process.py`—>`train.py`—>`predict.py`**</br>

### 评估
**脚本需要在linux下运行命令行 </br>
chmod a+x score.perl   ./score.perl 字典文件路径 标准分词结果路径 预测分词结果路径 >评估结果保存路径</br>
sighan2004语料,epoch=5的训练模型评估结果：</br>
2 Bi-LSTM layer + CRF & word embedding[50]** </br>
=== SUMMARY:</br>
=== TOTAL INSERTIONS:	2466</br>
=== TOTAL DELETIONS:	1733</br>
=== TOTAL SUBSTITUTIONS:	4238</br>
=== TOTAL NCHANGE:	8437</br>
=== TOTAL TRUE WORD COUNT:	106873</br>
=== TOTAL TEST WORD COUNT:	107606</br>
=== TOTAL TRUE WORDS RECALL:	0.944</br>
=== TOTAL TEST WORDS PRECISION:	0.938</br>
=== F MEASURE:	0.941</br>
=== OOV Rate:	0.026</br>
=== OOV Recall Rate:	0.667</br>
=== IV Recall Rate:	0.952</br>
=== test-pre.txt	2466	1733	4238	8437	106873	107606	0.944	0.938	0.941	0.026	0.667	0.952</br>
