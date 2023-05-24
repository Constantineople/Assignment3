# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:22:49 2022

@author: 123
"""
'''
宏观因子应该是多个大类资产的收益率（记为p个资产）中经过主成分分析
后这部分留下来的主成分（如中证500,10年期国债等）即构成了不同的
宏观因子
在学术以及实务中，估计因子暴露及因子收益主要有两种方法：
一种是时间序列回归，通过个股收益率序列对因子收益回归，估计因子暴露；
一种是横截面回归，在每一期通过个股收益率对因子暴露回归，估计因子收益率。
'''




import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut




logreturn_df = pd.read_excel("Logreturn.xlsx")
logreturn_df.dropna(inplace=(True))#删除缺失值
logreturn_df.columns.values.tolist()#获取列标签
XSHG_510300 = np.array(logreturn_df['510300.XSHG'])
XSHG_510500 = np.array(logreturn_df['510500.XSHG'])
XSHE_159920 = np.array(logreturn_df['159920.XSHE'])
XSHG_511010 = np.array(logreturn_df['511010.XSHG'])
XSHE_159934 = np.array(logreturn_df['159934.XSHE'])
XSHG_513100 = np.array(logreturn_df['513100.XSHG'])
XSHG_510050 = np.array(logreturn_df['510050.XSHG'])
logreturn = np.array([XSHG_510300,XSHG_510500,XSHE_159920,XSHG_511010,XSHE_159934,XSHG_513100,XSHG_510050])
#标准化之前要将df变成np.array
'''
这是以整个matrix标准化，并非按每一个列（即每一个资产）标准化
zs_scaler = preprocessing.StandardScaler()
standard_logreturn = zs_scaler.fit_transform(logreturn)#标准化收益率矩阵
'''

#这是最方便的按列标准化，每一列都减去该列的均值然后除以该列的std（axis = 0）说明了操作是对列进行的
logreturn_A = (logreturn - np.mean(logreturn, axis=0)) / np.std(logreturn, axis=0)
'''
因为我们计算的宏观因子是按照列（也就是按资产）来进行标准化的，这显然更合理
现在我们完成了第一步获取收益率矩阵并将其标准化
'''

'''
第二步，求解标准化收益率矩阵的相关系数矩阵
'''
logreturn_corr = np.corrcoef(logreturn_A)

'''
第三步要用雅克比方法计算相关系数矩阵的特征值以及相应的特征向量矩阵
'''
eigenvalue, featurevector = np.linalg.eig(logreturn_corr)
#featurevector即为特征向量矩阵（其每一列就是对应相关系数矩阵那列的特征向量）
'''
eigenvalue[ 2.70853631e+00, -6.85320603e-17,  
1.65562075e-01,  1.56153700e+00,1.21989365e+00,  8.27022010e-01,  5.17448963e-01]
510300:第一主成分，511010第二，159934第三，513100第四，510050第五，159920第六，510500第七
'''

'''
第四步降维后的主成分RF =特征矩阵E和初始矩阵（没标准化过的）X之间的线性组合
'''
#主成分是数据协方差矩阵的特征向量，矩阵乘其特征向量=对特征向量进行缩放
#比如3*2要变到3*1就要3*2×2*1，这个2*1就是其第一主成分的特征向量，即将数据投影到第一主成分上
#特征值大小可以判断谁是第一主成分

RF = np.dot(featurevector,logreturn)
#RF.shape为指数个行，交易日数个列
'''
特征矩阵featurevector代表着由原始收益矩阵转变为宏观因子收益矩阵的
线性组合系数。线性组合的不同，代表着降维后的主成分综合了大类资产中
某种不同的共性，从而使得主成分构成了不同的宏观风险因子。
RF现在是个turple，使用pd.DataFrame后再用shape可得到其大小是
7个资产行，1219个交易日列
求主成分的两种方法
1 从变量构成的矩阵X出发，先求出t(X)X的特征值和特征向量，
然后用X乘以特征向量就得到了主成份
2 从矩阵X的相关矩阵出发，求相关矩阵的特征值和特征向量，
然后用归一化的X乘以特征向量得到了主成份
'''

'''
直接回归
'''
weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
data_x = pd.DataFrame(RF.T)
#X是主成分每个交易日的值
portfolio = pd.read_excel('portfolio.xlsx')
IC = pd.DataFrame({})
for i in range(0,6):
    data_y = logreturn_df.iloc[:,2+i]
    res=np.linalg.lstsq(data_x,data_y,rcond=None)
    IC[logreturn_df.columns[2+i]] = res[0]
IC.to_excel('IC.xlsx')
#这里与纸上得到的不同，因为这里是进行了转置的
#纸上得到的IC的xij是第i个资产对第j个因子的暴露度，这里则是第j个资产对第i个因子（主成分）的暴露度
#Y是不同资产的每日收益率

#效果不好，换一种方法，用主成分分析做并做主成分回归

'''
#主成分PCA拟合，这里用了sklearn库
model = PCA()
model.fit(logreturn)
#每个主成分能解释的方差
model.explained_variance_
#每个主成分能解释的方差的百分比
model.explained_variance_ratio_
#可视化

plt.plot(model.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE')

plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')


X_train = logreturn_df.iloc[:610,2:]#1220个交易日，取前面一半训练
Y_train = portfolio.iloc[:610,2]
#0列是index,1,2,3,4,5...1列是Trade Days
X_test = logreturn_df.iloc[610:,2:]
Y_test = portfolio.iloc[610:,2]
#后面一半是测试集

scaler = StandardScaler()#进行标准化
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''
#留一法交叉验证选择误差最小的主成分个数
'''
scores_mse = []
for k in range(1, 7):
    model = PCA(n_components=k)
    model.fit(X_train)
    X_train_pca = model.transform(X_train)
    loo = LeaveOneOut()
    mse = -cross_val_score(LinearRegression(), X_train_pca, Y_train, 
                           cv=loo, scoring='neg_mean_squared_error')
    scores_mse.append(np.mean(mse))
min(scores_mse)
 
index = np.argmin(scores_mse)#误差最小时的主成分个数-1
index
 
plt.plot(range(1, 7), scores_mse)
plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('Leave-one-out Cross-validation Error')
plt.tight_layout()

#用上面得到的误差最小的主成分个数来进行回归
model = PCA(n_components = index + 1)
model.fit(X_train)
#得到主成分得分
X_train_pca = model.transform(X_train)
X_test_pca = model.transform(X_test)
X_train_pca
 
#进行线性回归拟合
reg = LinearRegression()
reg.fit(X_train_pca, Y_train)
 
#全样本预测
X_pca = np.vstack((X_train_pca, X_test_pca))
X_pca.shape
pred = reg.predict(X_pca)#预测值
 
y = portfolio.iloc[:, 2]
 
#可视化
plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.plot(y, label='Actual', color='r')
plt.plot(pred, label='Predicted', color='k', linestyle='--')
plt.xticks(range(1, 62))
ax.set_xticklabels(portfolio.index, rotation=90)
plt.axvline(610, color='k', linestyle='--', linewidth=1)
plt.xlabel('交易日')
plt.ylabel('收益率')
plt.title("回归")
plt.legend(loc='upper left')
plt.tight_layout()
'''