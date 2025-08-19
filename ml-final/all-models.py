# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from matplotlib.ticker import MultipleLocator
import os  # 导入 os 模块用于处理文件路径

# 加载数据
data = pd.read_csv(r"G:\N2-DG\new2\02-ML\ml-final\ML-data.csv")
X = data.iloc[:450, 1:6]
y = data.iloc[:450, 6]
print(f"数据形状: {X.shape}")

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 定义模型
models = {
    "Lasso": Lasso(alpha=0.000001, random_state=1),
    "GBR": GradientBoostingRegressor(alpha=0.30, n_estimators=320, learning_rate=0.10010136473684210526315, max_depth=5, min_samples_split=4, min_samples_leaf=3, subsample=0.7782, random_state=42),
    "GP": GaussianProcessRegressor(alpha=0.6, random_state=0),
    "DecisionTree": DecisionTreeRegressor(criterion='friedman_mse', max_depth=15, min_samples_split=5, min_samples_leaf=1, random_state=3),
    "RF": RandomForestRegressor(n_estimators=280, criterion='friedman_mse', min_samples_split=3, min_samples_leaf=1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=180, criterion='friedman_mse', max_depth=8, min_samples_split=5, min_samples_leaf=1, random_state=0),
    "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=7), n_estimators=28, random_state=42),
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
        n_jobs=None  # 设置为 None
    ),
    "XGBoost": xgb.XGBRegressor(base_score=0.182, booster='gbtree', colsample_bytree=0.85, gamma=0.0045031, learning_rate=0.100100136473684210526315, max_depth=5, min_child_weight=1, n_estimators=490, random_state=0, reg_alpha=0.15, reg_lambda=0.168, subsample=0.866469),
    "SVR": SVR(C=100, cache_size=24, coef0=0.0, degree=3, epsilon=0.05, gamma=6, kernel='rbf', max_iter=-1, shrinking=True, tol=0.1, verbose=False)
}

# 指定保存路径
save_path = r"G:\N2-DG\new2\02-ML\ml-final"
os.makedirs(save_path, exist_ok=True)  # 确保文件夹存在

# 训练模型并绘制散点图
for name, model in models.items():
    try:
        # 训练模型
        model.fit(X_train_std, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_std)
        y_pred_test = model.predict(X_test_std)
        
        # 绘制散点图
        fig, ax = plt.subplots(figsize=(6, 8))
        plt.axis([-30, 90, -30, 90])
        plt.xticks(size=18)
        plt.yticks(size=18)
        bwith = 3
        ax.spines["top"].set_linewidth(bwith)
        ax.spines["left"].set_linewidth(bwith)
        ax.spines["right"].set_linewidth(bwith)
        ax.spines["bottom"].set_linewidth(bwith)
        ax.tick_params(which='major', length=10, width=3)
        
        # 绘制训练集散点
        ax.scatter(y_pred_train, y_train, s=120, facecolors='none', edgecolors='green', linewidths=3, label="Training Set")
        # 绘制测试集散点
        ax.scatter(y_pred_test, y_test, color="red", s=120, label="Test Set")
        
        # 添加对角线
        line = plt.Line2D([0, 1], [0, 1], color='black', linewidth=3)
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        
        # 设置标签
        plt.xlabel(r'$\Delta{G}$-predictions (eV)', size=18, weight="bold")
        plt.ylabel(r'$\Delta{G}$-calculations (eV)', size=18, weight="bold")
        
        # 添加图例
        plt.legend(fontsize=18, loc="upper left", frameon=False)
        
        # 保存图表到指定路径
        plt.savefig(os.path.join(save_path, f'{name}_scatter.png'), bbox_inches='tight', dpi=300)
        
        # 显示图表
        plt.show()
        
        print(f"{name} 散点图绘制完成并保存到 {save_path}")
    except Exception as e:
        print(f"绘制 {name} 散点图时出错: {str(e)}")

# 汇总结果
results = []
results_train = []
for name, model in models.items():
    try:
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
    except Exception as e:
        print(f"计算 {name} 指标时出错: {str(e)}")

# 转换为DataFrame
results_df = pd.DataFrame(results)
results_df_train = pd.DataFrame(results_train)

# 打印结果表格
print("\n测试集性能指标结果:")
print(results_df[['Algorithm', 'RMSE', 'MAE', 'R2']].to_string(index=False))

print("\n训练集性能指标结果:")
print(results_df_train[['Algorithm', 'RMSE_train', 'MAE_train', 'R2_train']].to_string(index=False))