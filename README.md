# word_embedding
参考博客 http://linanqiu.github.io/2015/10/07/word2vec-sentiment/ 用gensim对doc2vec进行实践 （http://proceedings.mlr.press/v32/le14.pdf） ，数据来源于此博客。

doc2vec_imdb.py 主要对 imdb 影评（每个影评为一段话，视为doc，） 进行训练，直接得到每个doc的向量表示，此过程为非监督学习过程。然后，用简单的logitic regression训练已标注样本，并对未标注doc进行分类，从而实现情绪的positive和negative识别。

此外，word2vec_imdb.py 通过word2vec得到词向量（利用所有doc中的词进行训练）,每个文档向量用此文档中所有词向量的mean表示，最后用logistic regression实现情绪分类。

比较 直接得到 doc2vec 和 通过word2vec得到文档向量 间的差异。
