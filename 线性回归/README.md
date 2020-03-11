# 总结
## LinearRegression  
当特征与结果呈线性关系，噪声少的条件下使用。
## Ridge  
数据特征之间线性相关性较强（输入特征存在共线时），用LinearRegression类拟合的不是特别好，需要正则化，可以考虑用Ridge类。
## Lasso  
在一堆特征里面找出主要的特征（L1正则得到的解是一个稀疏解），那么Lasso回归便是首选，但是Lasso类需要对$\alpha$调优（同样Ridge类也是）。