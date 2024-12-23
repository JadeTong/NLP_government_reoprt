'''
自然语言处理NLP————2021年政府工作报告分析
    Python 自然语言处理中，英文处理常用的包是 NLTK 和 spaCy，中文处理常用的包是 jieba（中文分词工具）和 pynlpir（NLPIR 汉语分词系统），本节采用的是jieba。
    jieba（“结巴”中文分词工具）是目前较常用的中文文本处理系统。在jieba 包中，分词处理用到的函数为 jieba.posseg.cut()，自定义词江用到的函数为 jieba.add_word()，停用词处理建议自行编写代码，获得关键词用到的函数力 jieba.analyse.extract_tags()。 jieba 支持精确模式、全模式、搜索引擎模式和paddle模式4种分词模式，能够完成分词、添加自定义词典、关键词抽取、词性标注、返回词的位置等任务。
    本例采用jieba 进行文本数据分析，本例概述如下。
1.数据及分析对象
2021年3月5日在第十三届全国人民代表大会第四次会议上李克强总理所做的政府工作报告 。

2.目的及分析任务
分析2021年政府工作报告的关键内容。

3.方法及工具：jieba包
'''
#%%                       1.业务理解
'''
对第十三届全国人民代表大会第四次会议上的《政府工作报告》的文本内容进行自然语言处理和分析。
'''
#%%                       2.数据读取
text = open('D:/desktop/ML/自然语言处理/政府工作报告.txt', 'r', encoding='utf-8').read().replace('\n', '')  #'r'=in read mode, '.read()'将整个文件输入为一个string
text[:10] #查看前十个字符
#%%                       3.分词处理
#精准模式分词，将句子最精准地切开，适合文本分析，并输出词对应地词性。
import jieba
import jieba.posseg as pseg  #posseg用于需要词性标注
words = pseg.cut(text[:20]) #将文本前20个字符切割，并返回包含单词及其词性的对象。

for word, flag in words:     #遍历words，其中每个元素由分词结果 (word) 和对应的词性标注 (flag) 组成。
    print(F'{word} {flag}')  #使用 f-string 格式化字符串，打印每个分词结果及其词性。
# =============================================================================
# 各位 r
# 代表 n
# ： x
# 现在 t
# ， x
# 我 r
# 代表 n
# 国务院 nt
# ， x
# 向 p
# 大会 n
# 报告 n
# =============================================================================
# =============================================================================
# flag 是一个词性标签，标签如下：
# n	    普通名词	    f	  方位名词	   s	 处所名词        t     时间
# nr	人名	        ns	  地名     	   nt	 机构名          nw    作品名
# nz	其他专名	    v	  普通动词	   vd	 动副词          vn    名动词
# a	    形容词	    ad	  副形词	       an	 名形词          d     副词
# m	    数量词	    q	  量词	       r	 代词            p     介词
# c	    连词	        u	  助词	       xc	 其他虚词        w     标点符号
# PER	人名	        LOC   地名	       ORG	 机构名          TIME  时间
# =============================================================================
# 分词效果尚可，但系政府工作报告全文里面有很多专业术语，所以要添加自定义词汇后重新分词。

#%%                       4.添加自定义词汇
# 将“不平凡”“以保促稳”“稳中求进”“助企纾困”“量大面广”“中小微”“小微”“落实”“普惠”“稳岗”“线上”“放管服”“一带一路”“天问一号”“嫦娥五号”“因城施策”“线上”“不忘初心”“牢记使命”“探月”“可持续”“中国梦”添加到分词词典中，并为每个词标注对应的词性。
jieba.add_word('不平凡', tag='a')
jieba.add_word('以保促稳',tag='v' )
jieba.add_word('稳中求进',tag='v')
jieba.add_word('助企纾困',tag='v')
jieba.add_word('量大面广',tag='a')
jieba.add_word('中小微',tag='a')
jieba.add_word('小微',tag='a')
jieba.add_word('落实',tag='v')
jieba.add_word('普惠',tag='a')
jieba.add_word('稳岗',tag='v')
jieba.add_word('线上',tag='a')
jieba.add_word('放管服',tag='v')
jieba.add_word('一带一路',tag='n')
jieba.add_word('天间一号',tag='n')
jieba.add_word('嫦娥五号',tag='n')
jieba.add_word('因城施策',tag='v')
jieba.add_word('线上',tag='a')
jieba.add_word('不忘初心',tag='v')
jieba.add_word('牢记使命',tag='v')
jieba.add_word('探月',tag='n')
jieba.add_word('可持续',tag='a')
jieba.add_word('中国梦',tag='n')
jieba.add_word('做大',tag='v')
jieba.add_word('做优',tag='v')

# 再次采用精准模式分词，并查看最后20个字符的分词结果：
words = pseg.cut(text[-20:])
for word,flag in words:
    print(F'{word} {flag}')
# =============================================================================
# 、 x
# 实现 v
# 中华民族 nz
# 伟大 a
# 复兴 a
# 的 uj
# 中国梦 n
# 不懈 a
# 奋斗 v
# ！ x
# =============================================================================

#%%                       5.词性标注
# jieba的分词结果中包含了各个词的词性，将词、词性和数据年份存为列表，并查看前十条记录。
words = []
year = 2021
year_words = []
year_words.extend(pseg.cut(text)) #将分词结果扩展（extend）到 year_words 列表中。

words = [list(word) + [year] for word in year_words]
print(words[:5])

#%%% 将list转换为dataframe，并设置列名为‘词汇’、‘词性’、‘年份’
import pandas as pd
words = pd.DataFrame(words, columns=['词汇', '词性', '年份'])
# 根据词性中文表，将词性的中文名称也加上：
jiebapos = pd.read_excel('D:/desktop/ML/自然语言处理/jiebaPOS.xlsx', header=0)
jiebapos.rename(columns={'词性英文名称':'词性'}, inplace=True)
words = words.merge(jiebapos, how='left', on='词性')  #将两个表left join

words.isnull().any() #全表无缺失值

#%%                       6.停用词处理
# 切分后的词表中存在大量无意义的字词和标点符号，因此需要进行停用词处理。首先，读入停用词表stopword.txt，并查看前10个停用词字符：
stopwords = open('D:/desktop/ML/自然语言处理/stopwords.txt').read()
lst_StopWords = stopwords.split('\n')
lst_StopWords[:10]
# ['等', '都', '是', '而', '我', '这', '上', '就', '但', '给']

# 从 words 中过滤掉那些在 lst_StopWords 列表中的词汇
words = words[words.apply(lambda x:x.loc['词汇'] not in lst_StopWords, axis=1)]
words.shape
# 经过停用词处理的数据从9194条减少到7004条记录。

#%%                       7.词性分布分析
# 根据词性的中文名称统计词性出现的频数，查看出现频数最高的10个词性：
WordSpeechDistribution = pd.DataFrame(words['词性中文名称'].value_counts(ascending=False))
WordSpeechDistribution[:10]
#%%% 可视化
WordSpeechDistribution.rename(columns={'count':'频数'}, inplace=True)
WordSpeechDistribution.sort_values("频数", ascending=False)




import matplotlib.pyplot as plt
# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码问题
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.figure(figsize=(10, 6))
plt.bar(WordSpeechDistribution[:10].index, WordSpeechDistribution["频数"][:10], color="skyblue")
plt.title("前10词性频数分布", fontsize=16)
plt.xlabel("词性", fontsize=14)
plt.ylabel("频数", fontsize=14)
plt.show()


















