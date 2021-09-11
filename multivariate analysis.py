# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# a = pd.read_excel("./multivariate analysis/Pdata11_2.xlsx",header=None)
# b = a.values
# x0 = b[:-2,1:-1].astype(float)
# y0 = b[:-2,-1].astype(int)
# x = b[-2 :,1:-1]
# v = np.cov(x0.T)
# knn = KNeighborsClassifier(3,metric = 'mahalanobis',metric_params = {'V':v})
# knn.fit(x0,y0);pre = knn.predict(x);print("分类结果",pre)
# print("误判率:",1 - knn.score(x0,y0))
#Fisher准则
# import numpy as np
# from sklearn.discriminant_analysis import\
#     LinearDiscriminantAnalysis as LDA
# x0 = np.array([[1.24, 1.27], [1.36, 1.74],[1.38, 1.64],[1.38, 1.82],
#                [1.38, 1.90], [1.40, 1.70],[1.48, 1.82],[1.54, 1.82],[1.56, 2.08],
#                [1.14, 1.78],[1.18,1.96],[1.20,1.86],[1.26,2.00],[1.28, 2.00],[1.30,1.96]])
# x = np.array([[1.24, 1.80],[1.28,1.84],[1.40,2.04]])
# y0 = np.hstack([np.ones(9),2*np.ones(6)])
# clf = LDA()
# clf.fit(x0,y0)
# print("判别结果为:", clf.predict(x))
# print("已知样本的误判率为:", 1-clf.score(x0,y0))
#使用fisher准则去判断健康人群
# import pandas as pd
# from sklearn.discriminant_analysis import\
#     LinearDiscriminantAnalysis as LDA
# a = pd.read_excel("./multivariate analysis/Pdata11_2.xlsx",header=None)
# b = a.values
# x0 = b[:-2,1:-1].astype(float)
# y0 = b[:-2,-1].astype(int)
# x = b[-2:,1:-1]
# clf = LDA()
# clf.fit(x0,y0)
# print("判断结果为:",clf.predict(x))
# print("已知样本为错误率为:", 1-clf.score(x0,y0))
#贝叶斯判别法
# import numpy as np
# from sklearn.naive_bayes import GaussianNB
# x0 = np.array([[1.24, 1.27], [1.36, 1.74],[1.38, 1.64],[1.38, 1.82],
#                [1.38, 1.90], [1.40, 1.70],[1.48, 1.82],[1.54, 1.82],[1.56, 2.08],
#                 [1.14, 1.78],[1.18,1.96],[1.20,1.86],[1.26,2.00],[1.28, 2.00],[1.30,1.96]])
# x = np.array([[1.24,1.80], [1.28,1.84],[1.40, 2.04]])
# y0 = np.hstack([np.ones(9), 2*np.ones(6)])
# clf = GaussianNB()
# clf.fit(x0,y0)
# print("判别结果为:",clf.predict(x))
# print("已知样本误判率为:", 1-clf.score(x0,y0))
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import\
    LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
path = "./multivariate analysis/Pdata11_2.xlsx"
a = pd.read_excel(path, header = None)
b = a.values
x0 = b[:-2,1:-1].astype(float)
y0 = b[:-2,-1].astype(int)
model = LDA()
print(cross_val_score(model, x0,y0, cv =2))


