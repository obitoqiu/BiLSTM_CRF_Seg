# bi-lstm_crf_seg
## Use lstm and crf  in keras  to achieve Chinese word segement
---------------
### 简介
---------------
preprocess.py:      文件预处理（将分好词的训练集转换为三标签（B:词头 I:词中 S:单字词）序列标注的形式）</br>
</br>
process.py:         将文本序列化为模型输入、输出标准格式</br>
</br>
LSTM_model.py:      目前基本框架 embedding + bi_lstm + lstm + crf          </br>
                    词向量维度50，最长切分序列50，units=128，epoch=5...可调用model.summary()查看具体参数</br>
</br>
train.py:           训练模型</br>
</br>
predict.py:         生成预测文件</br>
</br>
test.py:            使用gensim生成简单的词向量</br>
</br>
-----------------
### 评估
-----------------
脚本需要在linux下运行 </br>
chmod a+x score.perl   ./score.perl 字典文件路径 标准分词结果路径 预测分词结果路径 >评估结果保存路径</br>
sighan2004语料,epoch=5的训练模型评估结果：</br>
=== SUMMARY:</br>
=== TOTAL INSERTIONS:	2535</br>
=== TOTAL DELETIONS:	2050</br>
=== TOTAL SUBSTITUTIONS:	4928</br>
=== TOTAL NCHANGE:	9513</br>
=== TOTAL TRUE WORD COUNT:	106873</br>
=== TOTAL TEST WORD COUNT:	107358</br>
=== TOTAL TRUE WORDS RECALL:	0.935</br>
=== TOTAL TEST WORDS PRECISION:	0.930</br>
=== F MEASURE:	0.933</br>
=== OOV Rate:	0.026</br>
=== OOV Recall Rate:	0.704</br>
=== IV Recall Rate:	0.941</br>
