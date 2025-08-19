# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker  # 导入刻度设置模块
from matplotlib.ticker import MultipleLocator
from sklearn import linear_model

# 加载数据
data = pd.read_csv(r"G:\N2-DG\new2\02-ML\ml-final\ML-data.csv")
X = data.iloc[:450, 1:6]
y = data.iloc[:450, 6]
print(f"数据形状: {X.shape}")
combined_kernel = Matern(length_scale=4.0, length_scale_bounds=(0.01, 30), nu=0.5) +RBF(length_scale=0.5)
# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 定义算法列表
models = {
    "Lasso": linear_model.Lasso(alpha=0.000001, copy_X=True, fit_intercept=True, max_iter=1000,
   positive=False, precompute=True, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=True),
    "GBR": ensemble.GradientBoostingRegressor(
        alpha=0.30, 
        n_estimators=320,          
        learning_rate=0.10010136473684210526315,         
        max_depth=5,               
        min_samples_split=4,       
        min_samples_leaf=3,        
        subsample=0.7782,             
        max_features=None,         
        random_state=42            
    ),
    "GP": GaussianProcessRegressor(combined_kernel,alpha=0.6,random_state=0),
    "DecisionTree": DecisionTreeRegressor(
        criterion='friedman_mse',  
        max_depth=15,               
        max_features=None,         
        max_leaf_nodes=None,       
        min_impurity_decrease=0.0, 
        min_samples_leaf=1,        
        min_samples_split=5,       
        min_weight_fraction_leaf=0.0, 
        random_state=3,            
        splitter='best'            
    ),
    "RF":  RandomForestRegressor(
        n_estimators=280,                
        criterion='friedman_mse',        
        max_depth=None,                  
        min_samples_split=3,             
        min_weight_fraction_leaf=0.0,    
        max_features=None,               
        max_leaf_nodes=None,             
        min_samples_leaf=1,              
        bootstrap=True,                  
        oob_score=False,                 
        n_jobs=1,                        
        min_impurity_decrease=0.0,       
        random_state=42,                 
        verbose=0,                       
        warm_start=False                 
    ),
    "ExtraTrees": ExtraTreesRegressor(
        bootstrap=False,                
        criterion='friedman_mse',       
        max_depth=8,                   
        max_features=None,              
        max_leaf_nodes=None,            
        min_impurity_decrease=0.0,      
        min_samples_leaf=1,             
        min_samples_split=5,            
        min_weight_fraction_leaf=0.0,   
        n_estimators=180,               
        n_jobs=1,                       
        oob_score=False,                
        random_state=0,                 
        verbose=0,                      
        warm_start=False                
    ),
    "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=7),
                            n_estimators=28, random_state=42),
    "LGBM": lgb.LGBMRegressor(
        boosting_type='gbdt',          
        num_leaves=26,                 
        max_depth=-1,                  
        learning_rate=0.1,             
        n_estimators=480,              
        subsample=0.85,                
        colsample_bytree=0.95,         
        reg_alpha=0.15,                
        reg_lambda=0.10,               
        random_state=42,               
    ),
    "XGBoost": xgb.XGBRegressor(base_score=0.182, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.85, gamma=0.0045031, importance_type='gain',
       learning_rate=0.100100136473684210526315, max_delta_step=0, max_depth=5,
       min_child_weight=1, n_estimators=490, n_jobs=1,
       nthread=None, objective='reg:linear', random_state=0, reg_alpha=0.15,
       reg_lambda=0.168, scale_pos_weight=1, subsample=0.866469),
    "SVR": SVR(C=100, cache_size=24, coef0=0.0, degree=3, epsilon=0.05, gamma=6, 
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.1, verbose=False)
}

# 训练模型并计算指标
results = []
results_train = []
for name, model in models.items():
    try:
        # 训练模型
        model.fit(X_train_std, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_std)
        y_pred_test = model.predict(X_test_std)
        
        # 计算测试集指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # 计算训练集指标
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        results.append({
            'Algorithm': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        results_train.append({
            'Algorithm': name,
            'RMSE_train': rmse_train,
            'MAE_train': mae_train,
            'R2_train': r2_train
        })
        print(f"{name} 训练完成")
    except Exception as e:
        print(f"训练 {name} 时出错: {str(e)}")

# 转换为DataFrame
results_df = pd.DataFrame(results)
results_df_train = pd.DataFrame(results_train)

# 按RMSE排序
results_df = results_df.sort_values(by='RMSE')
results_df_train = results_df_train.sort_values(by='RMSE_train')

# 绘制测试集性能比较图表
plt.figure(figsize=(7, 8))

# 创建主轴
ax1 = plt.subplot(111)
x = np.arange(len(results_df))
width = 0.35

# 绘制RMSE柱状图
ax1.bar(x - width/2, results_df['RMSE'], width, label='     ', color='skyblue')
#ax1.bar(x - width/2, results_df['RMSE'], width, label='RMSE', color='skyblue')
# 绘制MAE柱状图
ax1.bar(x + width/2, results_df['MAE'], width, label='    ', color='orange')
#ax1.bar(x + width/2, results_df['MAE'], width, label='MAE', color='orange')
# 设置主轴标签
#ax1.set_xlabel('机器学习算法', fontsize=18)
ax1.set_ylabel('Error (kcal/mol)', fontsize=18)
#ax1.set_title('测试集性能比较', fontsize=18)
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Algorithm'], rotation=90, ha='center', fontsize=18)
ax1.tick_params(axis='y', labelsize=18, width=3)
ax1.legend(fontsize=18, loc="upper left", frameon=False)
ax1.set_xlim(9.5, -0.5)
ax1.set_ylim(1, 15)
#ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 设置y轴最大标签个数为5
ax1.yaxis.set_major_locator(MultipleLocator(3))  # 设置左边Y轴间隔为1

# 创建次轴
ax2 = ax1.twinx()
# 绘制R2折线图
#ax2.plot(x, results_df['R2'], 'r*-', label='R$^2$',   markersize=18, linewidth=2,  color='red')
ax2.plot(x, results_df['R2'], 'r*-', label='     ',   markersize=18, linewidth=2,  color='red')
ax2.set_ylabel('R$^2$', fontsize=18)
ax2.legend(fontsize=18, loc="upper right", frameon=False)
ax2.set_ylim(0.5, 1.05)  # 设置右边Y轴范围为0到1
ax2.yaxis.set_major_locator(MultipleLocator(0.1))

# 设置次轴刻度标签大小
ax2.tick_params(axis='y', labelsize=18, width=3)
#ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 设置次轴y轴最大标签个数为5

# 设置次轴线宽
ax2.tick_params(width=3)
for spine in ax2.spines.values():
    spine.set_linewidth(3)

ax1.tick_params(which='major', length=10, width=3)
ax2.tick_params(which='major', length=10, width=3)


# 设置右边坐标轴和刻度为红色
ax2.spines['right'].set_color('red')
ax2.tick_params(axis='y', colors='red')

# 调整图例位置和间隔
#ax1.legend(bbox_to_anchor=(0.05, 0.95), fontsize=18, frameon=False)
#ax2.legend(bbox_to_anchor=(0.95, 0.95), fontsize=18, frameon=False)

# 调整布局
plt.subplots_adjust(hspace=10)  # 调整子图之间的间距
plt.tight_layout()

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig(r'test_performance.png', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()

# 打印结果表格
print("\n测试集性能指标结果:")
print(results_df[['Algorithm', 'RMSE', 'MAE', 'R2']].to_string(index=False))

# 绘制训练集性能比较图表
plt.figure(figsize=(15, 8))

# 创建主轴
ax3 = plt.subplot(111)
x = np.arange(len(results_df_train))
width = 0.35

# 绘制RMSE柱状图
ax3.bar(x - width/2, results_df_train['RMSE_train'], width, label='RMSE_train', color='skyblue')
# 绘制MAE柱状图
ax3.bar(x + width/2, results_df_train['MAE_train'], width, label='MAE_train', color='orange')

# 设置主轴标签
ax3.set_xlabel('机器学习算法', fontsize=18)
ax3.set_ylabel('RMSE_train/MAE_train', fontsize=18)
ax3.set_title('训练集性能比较', fontsize=18)
ax3.set_xticks(x)
ax3.set_xticklabels(results_df_train['Algorithm'], rotation=90, ha='center', fontsize=18)
ax3.tick_params(axis='y', labelsize=18, width=3)
ax3.legend(fontsize=18, loc="upper left", frameon=False)
ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 设置y轴最大标签个数为5

# 创建次轴
ax4 = ax3.twinx()
# 绘制R2折线图
ax4.plot(x, results_df_train['R2_train'], 'ro-', label='R2_train', markersize=18, color='green')
ax4.set_ylabel('R2_train', fontsize=18)
ax4.legend(fontsize=18, loc="upper right", frameon=False)

# 设置次轴刻度标签大小
ax4.tick_params(axis='y', labelsize=18, width=3)
ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 设置次轴y轴最大标签个数为5

# 设置次轴线宽
ax4.tick_params(width=3)
for spine in ax4.spines.values():
    spine.set_linewidth(3)

ax3.tick_params(which='major', length=10, width=3)
ax4.tick_params(which='major', length=10, width=3)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('train_performance.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

# 打印结果表格
print("\n训练集性能指标结果:")
print(results_df_train[['Algorithm', 'RMSE_train', 'MAE_train', 'R2_train']].to_string(index=False))