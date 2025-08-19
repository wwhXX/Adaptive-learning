# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from pylab import *
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from sklearn import ensemble
from sklearn import datasets
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xlwt
import math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVR,LinearSVR
from xgboost.sklearn import XGBRegressor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import entropy_sampling
from modAL.disagreement import max_std_sampling
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\External.csv")

n_lable = 450
n_initial = 450  #需要用多少数据进行模型训练
n_sample = 5467
mark=3
mark2=1
mark3=1
a=1000
X=data.iloc[:,1:6]
y=data.iloc[:,6]
x_index=data.iloc[:,7]
x_complex=data.iloc[:,0]
X_labeled=X[:n_initial]
print("X_labeled_data:", X_labeled.shape)
y_labeled=y[:n_initial]
y_labeled = pd.DataFrame(y_labeled)

labeled_indices = np.random.choice(X_labeled.shape[0], size=n_initial, replace=False)
print(labeled_indices.shape)
X_unlabeled =X[n_initial:n_sample] 
print("X_unlabeled_data:", X_unlabeled.shape)
y_unlabeled = y[n_initial:n_sample]
x_index_unlabeled=x_index[n_initial:n_sample]
x_index_info=x_complex[n_initial:n_sample]
n_repeats = 100
predictions_1st = []

seed_list = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\seed1.csv")
seed0 = seed_list.iloc[:, 0].tolist()
print(len(seed0))
for seed in seed0:

    model_xgboost = xgb.XGBRegressor(base_score=0.61, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.85, gamma=0.00045031, importance_type='gain',
       learning_rate=0.100136473684210526315, max_delta_step=0, max_depth=6,
       min_child_weight=1, n_estimators=280, n_jobs=1,
       nthread=None, objective='reg:linear', random_state=seed, reg_alpha=0.60,
       reg_lambda=0.38, scale_pos_weight=1, subsample=0.866445)
    
    learner = ActiveLearner(
    estimator=model_xgboost,
    X_training=X_labeled,
    y_training=y_labeled,
    )
    
    learner.fit(X_labeled,y_labeled)
    pred_train = learner.predict(X_labeled)
    y_pred = learner.predict(X_unlabeled)
    predictions_1st.append(y_pred)
    
    

list_shape = np.array(predictions_1st).shape


result_1st = pd.DataFrame(predictions_1st).T
result_1st.columns = [f'Prediction_{i+1}' for i in range(n_repeats)]
result_1st.to_csv('predictions_1st.csv', index=False)


pred_1st = result_1st.mean(axis=1)
std_1st = result_1st.std(axis=1)
pred_std_1st=np.vstack((x_index_info, x_index_unlabeled, pred_1st, y_unlabeled, std_1st, pred_1st-std_1st, pred_1st+std_1st)).T
pred_std_1st_list=pd.DataFrame(pred_std_1st)
pred_std_1st_list.columns = ["x_index_info", "x_index_unlabeled", "pred", "y_unlabeled","std","pred-std", "pred+std"]
pred_std_1st_list.to_csv("pred_std_1st.csv")

Al_1st = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\pred_std_1st.csv")

max_pred = Al_1st.nlargest(mark2, 'pred')
min_pred = Al_1st.nsmallest(mark3, 'pred')

largest_std = Al_1st.nlargest(mark, 'std')

Al1_MAE = pd.concat([max_pred, min_pred, largest_std], ignore_index=True)
Al1_MAE['absolute_deviation'] = np.abs(Al1_MAE['pred'] - Al1_MAE['y_unlabeled'])

Al1_MAE.to_csv("Al1_MAE.csv", index=False)

mean_absolute_deviation = np.mean(np.abs(Al1_MAE['pred'] - Al1_MAE['y_unlabeled']))

print(f"Al_1st平均偏差: {mean_absolute_deviation}")

Al1_MAE.to_csv("merged_data_with_Al1_MAE_deviation.csv", index=False)


mae_1st_train = metrics.mean_absolute_error(y_labeled, pred_train)
rmse_1st_train = math.sqrt(metrics.mean_squared_error(y_labeled, pred_train))
r2_1st_train = metrics.r2_score(y_labeled, pred_train)

print("第一轮模型训练的MAE：", mae_1st_train)
print("第一轮模型训练的RMSE：", rmse_1st_train)
print("第一轮模型训练的R2：", r2_1st_train)

fig, ax1 = plt.subplots(figsize=(20, 6))
plt.axis([n_initial - 20, n_sample + 20, -30, 60])
plt.xticks(size=18)
plt.yticks(size=18)
bwith = 3
ax1.spines["top"].set_linewidth(bwith)
ax1.spines["left"].set_linewidth(bwith)
ax1.spines["right"].set_linewidth(bwith)
ax1.spines["bottom"].set_linewidth(bwith)
ax1.tick_params(which='major', length=10, width=3)
plt.scatter(x_index_unlabeled, pred_1st, s=60, facecolors='none', edgecolors='grey', linewidths=3, label="Prediction")
plt.fill_between(x_index_unlabeled, pred_1st + std_1st, pred_1st - std_1st, facecolor='green', alpha=0.5, label="±1 Std Dev")
plt.legend(loc="best", fontsize=16)
plt.title('Prediction with Uncertainty', fontsize=20)
ax1.get_xaxis().set_visible(True)
ax1.grid(False)
plt.tight_layout()
plt.savefig('energy_AL1_with_std.png', bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(5, 6))
plt.axis([-30, 90, -30, 90])
plt.xticks(size=18)
plt.yticks(size=18)
bwith=3
ax.spines["top"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.spines["bottom"].set_linewidth(bwith)
ax.tick_params(which='major',length=10,width=3)
ax.scatter(pred_train, y_labeled,  s=100, facecolors='none',  edgecolors='green',linewidths=3, label=("1st-Training"))
ax.scatter(pred_1st, y_unlabeled, color="red", s=100, label="Test")


line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel(r'$\Delta{G}$-predictions', size=18, weight="bold")
plt.ylabel(r'$\Delta{G}$-predictions',size=18, weight="bold")
plt.legend()
plt.legend(fontsize=18,loc="upper left",frameon=False)
plt.savefig('energy-AL1.png', bbox_inches='tight', dpi=300)
plt.show()


pred_1st=pd.DataFrame(pred_1st)
std_1st=pd.DataFrame(std_1st)
print("std_1st:",std_1st)
x_index_unlabeled=pd.DataFrame(x_index_unlabeled)
std_1st=np.hstack((x_index_unlabeled, std_1st,pred_1st))
std_1st=pd.DataFrame(std_1st)
std_1st = std_1st.rename(columns={
    std_1st.columns[0]: "query_indices",
    std_1st.columns[1]: "std_1st", 
    std_1st.columns[2]: "pred_1st"
    })

std_1st["std_1st"] = std_1st["std_1st"].astype(float)
std_1st["pred_1st"] = std_1st["pred_1st"].astype(float)

std_1st.to_csv('std_1st.csv', index=False)

top_mark_std_1st = std_1st.nlargest(mark, 'std_1st')
print(top_mark_std_1st)
query_indices_std_1st=top_mark_std_1st['query_indices']
query_indices_std_1st=pd.DataFrame(query_indices_std_1st) 


top_mark_pred_1st_min = std_1st.nsmallest(mark2, 'pred_1st')
print(top_mark_pred_1st_min)
query_indices_pred_1st_min=top_mark_pred_1st_min['query_indices']
query_indices_pred_1st_min=pd.DataFrame(query_indices_pred_1st_min) 

top_mark_pred_1st_max = std_1st.nlargest(mark3, 'pred_1st')
print(top_mark_pred_1st_max)
query_indices_pred_1st_max=top_mark_pred_1st_max ['query_indices']
query_indices_pred_1st_max=pd.DataFrame(query_indices_pred_1st_max) 


query_indices=pd.concat([query_indices_std_1st, query_indices_pred_1st_min,query_indices_pred_1st_max])

query_indices = query_indices.squeeze()
query_indices = query_indices.astype(int)
query_indices = np.array(query_indices)
print(query_indices)
print(query_indices.shape)

new_labeled_X = X.iloc[query_indices, :]
new_labeled_y = y[query_indices]
new_labeled_y = pd.DataFrame(new_labeled_y)
print("1st_new_labeled_y:", new_labeled_y)

labeled_indices = np.append(labeled_indices, query_indices)
X_labeled = X.iloc[labeled_indices, :]
y_labeled = y.iloc[labeled_indices]

unlabeled_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indices)
print("unlabeled_indices:", unlabeled_indices.shape)
X_unlabeled = X.iloc[unlabeled_indices, :]
y_unlabeled = y.iloc[unlabeled_indices]
x_index_unlabeled = x_index.iloc[unlabeled_indices]
x_index_info=x_complex[unlabeled_indices]

predictions_2nd =[]
for seed in seed0:  
    learner.teach(new_labeled_X, new_labeled_y)
    
    pred_train = learner.predict(X_labeled)
    y_pred = learner.predict(X_unlabeled)
    predictions_2nd.append(y_pred)



list_shape = np.array(predictions_2nd).shape

result_2nd = pd.DataFrame(predictions_2nd).T
result_2nd.columns = [f'Prediction_{i+1}' for i in range(n_repeats)]

result_2nd.to_csv('predictions_2nd.csv', index=False)
pred_2nd = result_2nd.mean(axis=1)


std_2nd = result_2nd.std(axis=1)

pred_std_2nd=np.vstack((x_index_info, x_index_unlabeled, pred_2nd, y_unlabeled,std_2nd, pred_2nd-std_2nd, pred_2nd+std_2nd)).T
pred_std_2nd_list=pd.DataFrame(pred_std_2nd)
pred_std_2nd_list.columns = ["x_index_info", "x_index_unlabeled","pred", "y_unlabeled","std","pred-std", "pred+std"]
pred_std_2nd_list.to_csv("pred_std_2nd.csv")



mae_2nd_train = metrics.mean_absolute_error(y_labeled, pred_train)
rmse_2nd_train = math.sqrt(metrics.mean_squared_error(y_labeled, pred_train))
r2_2nd_train = metrics.r2_score(y_labeled, pred_train)

print("第二轮模型训练的MAE：", mae_2nd_train)
print("第二轮模型训练的RMSE：", rmse_2nd_train)
print("第二轮模型训练的R2：", r2_2nd_train)

Al_2nd = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\pred_std_2nd.csv")



largest_std = Al_2nd.nlargest(mark, 'std')

Al_2nd = Al_2nd[~Al_2nd.index.isin(largest_std.index)]

max_pred = Al_2nd.nlargest(mark2, 'pred')
print("max_pred:", max_pred)

Al_2nd = Al_2nd[~Al_2nd.index.isin(max_pred.index)]

min_pred = Al_2nd.nsmallest(mark3, 'pred')
print("min_pred:", min_pred)

Al2_MAE = pd.concat([max_pred, min_pred, largest_std], ignore_index=True)
Al2_MAE['absolute_deviation'] = np.abs(Al2_MAE['pred'] - Al2_MAE['y_unlabeled'])

print("合并后的数据：", Al2_MAE)

Al2_MAE.to_csv("Al2_MAE.csv", index=False)
mean_absolute_deviation = np.mean(np.abs(Al2_MAE['pred'] - Al2_MAE['y_unlabeled']))
print(f"Al2_MAE平均偏差: {mean_absolute_deviation}")

Al2_MAE.to_csv("merged_data_with_Al2_MAE_deviation.csv", index=False)


fig, ax1 = plt.subplots(figsize=(20, 6))

plt.axis([n_initial - 20, n_sample + 20, -30, 60])

plt.xticks(size=18)
plt.yticks(size=18)

bwith = 3
ax1.spines["top"].set_linewidth(bwith)
ax1.spines["left"].set_linewidth(bwith)
ax1.spines["right"].set_linewidth(bwith)
ax1.spines["bottom"].set_linewidth(bwith)

ax1.tick_params(which='major', length=10, width=3)

plt.scatter(x_index_unlabeled, pred_2nd, s=60, facecolors='none', edgecolors='grey', linewidths=3, label="Prediction")

plt.fill_between(x_index_unlabeled, pred_2nd+std_2nd, pred_2nd-std_2nd, facecolor='green', alpha=0.5, label="±1 Std Dev")

plt.legend(loc="best", fontsize=16)

plt.title('Prediction with Uncertainty-2nd', fontsize=20)

ax1.get_xaxis().set_visible(True)
ax1.grid(False)
plt.tight_layout()
plt.savefig('energy-AL2-std.png', bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(5, 6))
plt.axis([-30, 90, -30, 90])
plt.xticks(size=18)
plt.yticks(size=18)
bwith=3
ax.spines["top"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.spines["bottom"].set_linewidth(bwith)
ax.tick_params(which='major',length=10,width=3)
ax.scatter(pred_train, y_labeled,  s=100, facecolors='none',  edgecolors='green',linewidths=3, label=("2nd-Training"))
ax.scatter(pred_2nd, y_unlabeled, color="red", s=100, label="Test")


line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel(r'$\Delta{G}$-predictions', size=18, weight="bold")
plt.ylabel(r'$\Delta{G}$-predictions',size=18, weight="bold")
plt.legend()
plt.legend(fontsize=18,loc="upper left",frameon=False)
plt.savefig('energy-AL2.png', bbox_inches='tight', dpi=300)
plt.show()

pred_2nd=pd.DataFrame(pred_2nd)
std_2nd=pd.DataFrame(std_2nd)
x_index_unlabeled=pd.DataFrame(x_index_unlabeled)
std_2nd=np.hstack((x_index_unlabeled, std_2nd, pred_2nd))
std_2nd=pd.DataFrame(std_2nd)
std_2nd = std_2nd.rename(columns={
    std_2nd.columns[0]: "query_indices", 
    std_2nd.columns[1]: "std_2nd",
    std_2nd.columns[2]: "pred_2nd"
    })

std_2nd["std_2nd"] = std_2nd["std_2nd"].astype(float)
std_2nd["pred_2nd"] = std_2nd["pred_2nd"].astype(float)

std_2nd.to_csv('std_2nd.csv', index=False)


top_mark_std_2nd = std_2nd.nlargest(mark, 'std_2nd')
print("top_mark_std_2nd:", top_mark_std_2nd)
query_indices_std_2nd = top_mark_std_2nd['query_indices']
query_indices_std_2nd = pd.DataFrame(query_indices_std_2nd)


top_mark_pred_2nd_min = std_2nd.nsmallest(mark2, 'pred_2nd')
print("top_mark_pred_2nd_min:", top_mark_pred_2nd_min)
query_indices_pred_2nd_min = top_mark_pred_2nd_min['query_indices']
query_indices_pred_2nd_min = pd.DataFrame(query_indices_pred_2nd_min)

while len(set(query_indices_std_2nd['query_indices']).intersection(set(query_indices_pred_2nd_min['query_indices']))) > 0:

    std_2nd = std_2nd[~std_2nd['query_indices'].isin(query_indices_std_2nd['query_indices'])]

    top_mark_pred_2nd_min = std_2nd.nsmallest(mark2, 'pred_2nd')
    print("Re-finding top_mark_pred_2nd_min:", top_mark_pred_2nd_min)
    query_indices_pred_2nd_min = top_mark_pred_2nd_min['query_indices']
    query_indices_pred_2nd_min = pd.DataFrame(query_indices_pred_2nd_min)

top_mark_pred_2nd_max = std_2nd.nlargest(mark3, 'pred_2nd')
print("top_mark_pred_2nd_max:", top_mark_pred_2nd_max)
query_indices_pred_2nd_max = top_mark_pred_2nd_max['query_indices']
query_indices_pred_2nd_max = pd.DataFrame(query_indices_pred_2nd_max)

while len(set(query_indices_std_2nd['query_indices']).intersection(set(query_indices_pred_2nd_max['query_indices']))) > 0:

    std_2nd = std_2nd[~std_2nd['query_indices'].isin(query_indices_std_2nd['query_indices'])]

    top_mark_pred_2nd_max = std_2nd.nlargest(mark3, 'pred_2nd')
    print("Re-finding top_mark_pred_2nd_max:", top_mark_pred_2nd_max)
    query_indices_pred_2nd_max = top_mark_pred_2nd_max['query_indices']
    query_indices_pred_2nd_max = pd.DataFrame(query_indices_pred_2nd_max)


query_indices=pd.concat([query_indices_std_2nd, query_indices_pred_2nd_min,query_indices_pred_2nd_max])


query_indices = query_indices.squeeze()
query_indices = query_indices.astype(int)
query_indices = np.array(query_indices)
print(query_indices)
print(query_indices.shape)
new_labeled_X = X.iloc[query_indices, :]

new_labeled_y = y[query_indices]
new_labeled_y = pd.DataFrame(new_labeled_y)
print("2nd_new_labeled_y:", new_labeled_y)

labeled_indices = np.append(labeled_indices, query_indices)
X_labeled = X.iloc[labeled_indices, :]
y_labeled = y.iloc[labeled_indices]



unlabeled_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indices)
print("unlabeled_indices:", unlabeled_indices.shape)
X_unlabeled = X.iloc[unlabeled_indices, :]
y_unlabeled = y.iloc[unlabeled_indices]
x_index_unlabeled = x_index.iloc[unlabeled_indices]
x_index_info=x_complex[unlabeled_indices]


predictions_3rd =[]
for seed in seed0: 
    learner.teach(new_labeled_X, new_labeled_y)
    
    pred_train = learner.predict(X_labeled)
    y_pred = learner.predict(X_unlabeled)
    predictions_3rd.append(y_pred)



list_shape = np.array(predictions_3rd).shape
result_3rd = pd.DataFrame(predictions_3rd).T
result_3rd.columns = [f'Prediction_{i+1}' for i in range(n_repeats)]
result_3rd.to_csv('predictions_3rd.csv', index=False)
pred_3rd = result_3rd.mean(axis=1)

std_3rd = result_3rd.std(axis=1)

pred_std_3rd=np.vstack((x_index_info, x_index_unlabeled, pred_3rd, y_unlabeled,std_3rd, pred_3rd-std_3rd, pred_3rd+std_3rd)).T
pred_std_3rd_list=pd.DataFrame(pred_std_3rd)
pred_std_3rd_list.columns = ["x_index_info", "x_index_unlabeled","pred", "y_unlabeled","std","pred-std", "pred+std"]
pred_std_3rd_list.to_csv("pred_std_3rd.csv")

Al_3rd = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\pred_std_3rd.csv")


max_pred = Al_3rd.nlargest(mark2, 'pred')
min_pred = Al_3rd.nsmallest(mark3, 'pred')

largest_std = Al_3rd.nlargest(mark, 'std')


Al3_MAE = pd.concat([max_pred, min_pred, largest_std], ignore_index=True)
Al3_MAE['absolute_deviation'] = np.abs(Al3_MAE['pred'] - Al3_MAE['y_unlabeled'])

Al3_MAE.to_csv("Al3_MAE.csv", index=False)


mean_absolute_deviation = np.mean(np.abs(Al3_MAE['pred'] - Al3_MAE['y_unlabeled']))

print(f"Al3_MAE平均偏差: {mean_absolute_deviation}")
Al3_MAE.to_csv("merged_data_with_Al3_MAE_deviation.csv", index=False)

mae_3rd_train = metrics.mean_absolute_error(y_labeled, pred_train)
rmse_3rd_train = math.sqrt(metrics.mean_squared_error(y_labeled, pred_train))
r2_3rd_train = metrics.r2_score(y_labeled, pred_train)

print("第三轮模型训练的MAE：", mae_3rd_train)
print("第三轮模型训练的RMSE：", rmse_3rd_train)
print("第三轮模型训练的R2：", r2_3rd_train)

fig, ax1 = plt.subplots(figsize=(20, 6))

plt.axis([n_initial - 20, n_sample + 20, -30, 60])

plt.xticks(size=18)
plt.yticks(size=18)

bwith = 3
ax1.spines["top"].set_linewidth(bwith)
ax1.spines["left"].set_linewidth(bwith)
ax1.spines["right"].set_linewidth(bwith)
ax1.spines["bottom"].set_linewidth(bwith)

ax1.tick_params(which='major', length=10, width=3)

plt.scatter(x_index_unlabeled, pred_3rd, s=60, facecolors='none', edgecolors='grey', linewidths=3, label="Prediction")
plt.fill_between(x_index_unlabeled, pred_3rd+std_3rd, pred_3rd-std_3rd, facecolor='green', alpha=0.5, label="±1 Std Dev")
plt.legend(loc="best", fontsize=16)
plt.title('Prediction with Uncertainty-3rd', fontsize=20)
ax1.get_xaxis().set_visible(True)
ax1.grid(False)
plt.tight_layout()
plt.savefig('energy-AL3-std.png', bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(5, 6))
plt.axis([-30, 90, -30, 90])
plt.xticks(size=18)
plt.yticks(size=18)
bwith=3
ax.spines["top"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.spines["bottom"].set_linewidth(bwith)
ax.tick_params(which='major',length=10,width=3)
ax.scatter(pred_train, y_labeled,  s=100, facecolors='none',  edgecolors='green',linewidths=3, label=("3rd-Training"))
ax.scatter(pred_3rd, y_unlabeled, color="red", s=100, label="Test")


line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel(r'$\Delta{G}$-predictions', size=18, weight="bold")
plt.ylabel(r'$\Delta{G}$-predictions',size=18, weight="bold")
plt.legend()
plt.legend(fontsize=18,loc="upper left",frameon=False)
plt.savefig('energy-AL3.png', bbox_inches='tight', dpi=300)
plt.show()


pred_3rd=pd.DataFrame(pred_3rd)
std_3rd=pd.DataFrame(std_3rd)
x_index_unlabeled=pd.DataFrame(x_index_unlabeled)
std_3rd=np.hstack((x_index_unlabeled, std_3rd, pred_3rd))
std_3rd=pd.DataFrame(std_3rd)
std_3rd = std_3rd.rename(columns={
    std_3rd.columns[0]: "query_indices", 
    std_3rd.columns[1]: "std_3rd",
    std_3rd.columns[2]: "pred_3rd"
    })

std_3rd["std_3rd"] = std_3rd["std_3rd"].astype(float)
std_3rd["pred_3rd"] = std_3rd["pred_3rd"].astype(float)

std_3rd.to_csv('std_3rd.csv', index=False)
top_mark_std_3rd = std_3rd.nlargest(mark, 'std_3rd')
print("top_mark_std_3rd:", top_mark_std_3rd)
query_indices_std_3rd = top_mark_std_3rd['query_indices']
query_indices_std_3rd = pd.DataFrame(query_indices_std_3rd)

top_mark_pred_3rd_min = std_3rd.nsmallest(mark2, 'pred_3rd')
print("top_mark_pred_3rd_min:", top_mark_pred_3rd_min)
query_indices_pred_3rd_min = top_mark_pred_3rd_min['query_indices']
query_indices_pred_3rd_min = pd.DataFrame(query_indices_pred_3rd_min)

while len(set(query_indices_std_3rd['query_indices']).intersection(set(query_indices_pred_3rd_min['query_indices']))) > 0:
    std_3rd = std_3rd[~std_3rd['query_indices'].isin(query_indices_std_3rd['query_indices'])]
    top_mark_pred_3rd_min = std_3rd.nsmallest(mark2, 'pred_3rd')
    print("Re-finding top_mark_pred_3rd_min:", top_mark_pred_3rd_min)
    query_indices_pred_3rd_min = top_mark_pred_3rd_min['query_indices']
    query_indices_pred_3rd_min = pd.DataFrame(query_indices_pred_3rd_min)

top_mark_pred_3rd_max = std_3rd.nlargest(mark3, 'pred_3rd')
print("top_mark_pred_3rd_max:", top_mark_pred_3rd_max)
query_indices_pred_3rd_max = top_mark_pred_3rd_max['query_indices']
query_indices_pred_3rd_max = pd.DataFrame(query_indices_pred_3rd_max)

while len(set(query_indices_std_3rd['query_indices']).intersection(set(query_indices_pred_3rd_max['query_indices']))) > 0:
    std_3rd = std_3rd[~std_3rd['query_indices'].isin(query_indices_std_3rd['query_indices'])]
    top_mark_pred_3rd_max = std_3rd.nlargest(mark3, 'pred_3rd')
    print("Re-finding top_mark_pred_3rd_max:", top_mark_pred_3rd_max)
    query_indices_pred_3rd_max = top_mark_pred_3rd_max['query_indices']
    query_indices_pred_3rd_max = pd.DataFrame(query_indices_pred_3rd_max)

query_indices=pd.concat([query_indices_std_3rd, query_indices_pred_3rd_min,query_indices_pred_3rd_max])

query_indices = query_indices.squeeze()
query_indices = query_indices.astype(int)
query_indices = np.array(query_indices)
print(query_indices)
print(query_indices.shape)
new_labeled_X = X.iloc[query_indices, :]
new_labeled_y = y[query_indices]
new_labeled_y = pd.DataFrame(new_labeled_y)
print("3rd_new_labeled_y:", new_labeled_y)
labeled_indices = np.append(labeled_indices, query_indices)
X_labeled = X.iloc[labeled_indices, :]
y_labeled = y.iloc[labeled_indices]
unlabeled_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indices)
print("unlabeled_indices:", unlabeled_indices.shape)
X_unlabeled = X.iloc[unlabeled_indices, :]
y_unlabeled = y.iloc[unlabeled_indices]
x_index_unlabeled = x_index.iloc[unlabeled_indices]
x_index_info=x_complex[unlabeled_indices]

predictions_4th =[]
for seed in seed0:
  
    learner.teach(new_labeled_X, new_labeled_y)
    
    pred_train = learner.predict(X_labeled)
    y_pred = learner.predict(X_unlabeled)
    predictions_4th.append(y_pred)
list_shape = np.array(predictions_4th).shape
result_4th = pd.DataFrame(predictions_4th).T
result_4th.columns = [f'Prediction_{i+1}' for i in range(n_repeats)]
result_4th.to_csv('predictions_4th.csv', index=False)

pred_4th = result_4th.mean(axis=1)

std_4th = result_4th.std(axis=1)

pred_std_4th=np.vstack((x_index_info, x_index_unlabeled, pred_4th, y_unlabeled,std_4th, pred_4th-std_4th, pred_4th+std_4th)).T

pred_std_4th_list=pd.DataFrame(pred_std_4th)
pred_std_4th_list.columns = ["x_index_info", "x_index_unlabeled","pred", "y_unlabeled","std","pred-std", "pred+std"]
pred_std_4th_list.to_csv("pred_std_4th.csv")

Al_4th = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\pred_std_4th.csv")

max_pred = Al_4th.nlargest(mark2, 'pred')
min_pred = Al_4th.nsmallest(mark3, 'pred')
largest_std = Al_4th.nlargest(mark, 'std')
Al4_MAE = pd.concat([max_pred, min_pred, largest_std], ignore_index=True)
Al4_MAE['absolute_deviation'] = np.abs(Al4_MAE['pred'] - Al4_MAE['y_unlabeled'])
Al4_MAE.to_csv("Al4_MAE.csv", index=False)


mean_absolute_deviation = np.mean(np.abs(Al4_MAE['pred'] - Al4_MAE['y_unlabeled']))

print(f"Al4_MAE平均偏差: {mean_absolute_deviation}")
Al4_MAE.to_csv("merged_data_with_Al4_MAE_deviation.csv", index=False)

mae_4th_train = metrics.mean_absolute_error(y_labeled, pred_train)
rmse_4th_train = math.sqrt(metrics.mean_squared_error(y_labeled, pred_train))
r2_4th_train = metrics.r2_score(y_labeled, pred_train)

print("第四轮模型训练的MAE：", mae_4th_train)
print("第四轮模型训练的RMSE：", rmse_4th_train)
print("第四轮模型训练的R2：", r2_4th_train)

fig, ax1 = plt.subplots(figsize=(20, 6))
plt.axis([n_initial - 20, n_sample + 20, -30, 60])
plt.xticks(size=18)
plt.yticks(size=18)
bwith = 3
ax1.spines["top"].set_linewidth(bwith)
ax1.spines["left"].set_linewidth(bwith)
ax1.spines["right"].set_linewidth(bwith)
ax1.spines["bottom"].set_linewidth(bwith)
ax1.tick_params(which='major', length=10, width=3)
plt.scatter(x_index_unlabeled, pred_4th, s=60, facecolors='none', edgecolors='grey', linewidths=3, label="Prediction")
plt.fill_between(x_index_unlabeled, pred_4th+std_4th, pred_4th-std_4th, facecolor='green', alpha=0.5, label="±1 Std Dev")
plt.legend(loc="best", fontsize=16)
plt.title('Prediction with Uncertainty-4th', fontsize=20)
ax1.get_xaxis().set_visible(True)
ax1.grid(False)
plt.tight_layout()
plt.savefig('energy-AL4-std.png', bbox_inches='tight', dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(5, 6))
plt.axis([-30, 90, -30, 90])
plt.xticks(size=18)
plt.yticks(size=18)
bwith=3
ax.spines["top"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.spines["bottom"].set_linewidth(bwith)
ax.tick_params(which='major',length=10,width=3)
ax.scatter(pred_train, y_labeled,  s=100, facecolors='none',  edgecolors='green',linewidths=3, label=("4th-Training"))
ax.scatter(pred_4th, y_unlabeled, color="red", s=100, label="Test")

line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel(r'$\Delta{G}$-predictions', size=18, weight="bold")
plt.ylabel(r'$\Delta{G}$-predictions',size=18, weight="bold")
plt.legend()
plt.legend(fontsize=18,loc="upper left",frameon=False)
plt.savefig('energy-AL4.png', bbox_inches='tight', dpi=300)
plt.show()

pred_4th=pd.DataFrame(pred_4th)
std_4th=pd.DataFrame(std_4th)
x_index_unlabeled=pd.DataFrame(x_index_unlabeled)
std_4th=np.hstack((x_index_unlabeled, std_4th, pred_4th))
std_4th=pd.DataFrame(std_4th)
std_4th = std_4th.rename(columns={
    std_4th.columns[0]: "query_indices", 
    std_4th.columns[1]: "std_4th",
    std_4th.columns[2]: "pred_4th"
    })
std_4th["std_4th"] = std_4th["std_4th"].astype(float)
std_4th["pred_4th"] = std_4th["pred_4th"].astype(float)

std_4th.to_csv('std_4th.csv', index=False)

top_mark_std_4th = std_4th.nlargest(mark, 'std_4th')
print("top_mark_std_4th:", top_mark_std_4th)
query_indices_std_4th = top_mark_std_4th['query_indices']
query_indices_std_4th = pd.DataFrame(query_indices_std_4th)

top_mark_pred_4th_min = std_4th.nsmallest(mark2, 'pred_4th')
print("top_mark_pred_4th_min:", top_mark_pred_4th_min)
query_indices_pred_4th_min = top_mark_pred_4th_min['query_indices']
query_indices_pred_4th_min = pd.DataFrame(query_indices_pred_4th_min)

while len(set(query_indices_std_4th['query_indices']).intersection(set(query_indices_pred_4th_min['query_indices']))) > 0:
    std_4th = std_4th[~std_4th['query_indices'].isin(query_indices_std_4th['query_indices'])]
    top_mark_pred_4th_min = std_4th.nsmallest(mark2, 'pred_4th')
    print("Re-finding top_mark_pred_4th_min:", top_mark_pred_4th_min)
    query_indices_pred_4th_min = top_mark_pred_4th_min['query_indices']
    query_indices_pred_4th_min = pd.DataFrame(query_indices_pred_4th_min)
top_mark_pred_4th_max = std_4th.nlargest(mark3, 'pred_4th')
print("top_mark_pred_4th_max:", top_mark_pred_4th_max)
query_indices_pred_4th_max = top_mark_pred_4th_max['query_indices']
query_indices_pred_4th_max = pd.DataFrame(query_indices_pred_4th_max)
while len(set(query_indices_std_4th['query_indices']).intersection(set(query_indices_pred_4th_max['query_indices']))) > 0:
    std_4th = std_4th[~std_4th['query_indices'].isin(query_indices_std_4th['query_indices'])]
    top_mark_pred_4th_max = std_4th.nlargest(mark3, 'pred_4th')
    print("Re-finding top_mark_pred_4th_max:", top_mark_pred_4th_max)
    query_indices_pred_4th_max = top_mark_pred_4th_max['query_indices']
    query_indices_pred_4th_max = pd.DataFrame(query_indices_pred_4th_max)


query_indices=pd.concat([query_indices_std_4th, query_indices_pred_4th_min,query_indices_pred_4th_max])
query_indices = query_indices.squeeze()
query_indices = query_indices.astype(int)
query_indices = np.array(query_indices)
print(query_indices)
print(query_indices.shape)

new_labeled_X = X.iloc[query_indices, :]
new_labeled_y = y[query_indices]
new_labeled_y = pd.DataFrame(new_labeled_y)
print("4th_new_labeled_y:", new_labeled_y)

labeled_indices = np.append(labeled_indices, query_indices)
X_labeled = X.iloc[labeled_indices, :]
y_labeled = y.iloc[labeled_indices]
unlabeled_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indices)
print("unlabeled_indices:", unlabeled_indices.shape)
X_unlabeled = X.iloc[unlabeled_indices, :]
y_unlabeled = y.iloc[unlabeled_indices]
x_index_unlabeled = x_index.iloc[unlabeled_indices]
x_index_info=x_complex[unlabeled_indices]

predictions_5th =[]
for seed in seed0:
    learner.teach(new_labeled_X, new_labeled_y)
    
    pred_train = learner.predict(X_labeled)
    y_pred = learner.predict(X_unlabeled)
    predictions_5th.append(y_pred)

list_shape = np.array(predictions_5th).shape
result_5th = pd.DataFrame(predictions_5th).T
result_5th.columns = [f'Prediction_{i+1}' for i in range(n_repeats)]
result_5th.to_csv('predictions_5th.csv', index=False)

pred_5th = result_5th.mean(axis=1)

std_5th = result_5th.std(axis=1)
pred_std_5th=np.vstack((x_index_info, x_index_unlabeled, pred_5th, y_unlabeled,std_5th, pred_5th-std_5th, pred_5th+std_5th)).T
pred_std_5th_list=pd.DataFrame(pred_std_5th)
pred_std_5th_list.columns = ["x_index_info", "x_index_unlabeled","pred", "y_unlabeled","std","pred-std", "pred+std"]
pred_std_5th_list.to_csv("pred_std_5th.csv")

Al_5th = pd.read_csv(r"G:\N2-DG\new2\03-AL\final-AL\pred_std_5th.csv")
max_pred = Al_5th.nlargest(mark2, 'pred')
min_pred = Al_5th.nsmallest(mark3, 'pred')
largest_std = Al_5th.nlargest(mark, 'std')

Al5_MAE = pd.concat([max_pred, min_pred, largest_std], ignore_index=True)
Al5_MAE['absolute_deviation'] = np.abs(Al5_MAE['pred'] - Al5_MAE['y_unlabeled'])
Al5_MAE.to_csv("Al5_MAE.csv", index=False)
mean_absolute_deviation = np.mean(np.abs(Al5_MAE['pred'] - Al5_MAE['y_unlabeled']))

print(f"Al5_MAE平均偏差: {mean_absolute_deviation}")
Al5_MAE.to_csv("merged_data_with_Al5_MAE_deviation.csv", index=False)
mae_5th_train = metrics.mean_absolute_error(y_labeled, pred_train)
rmse_5th_train = math.sqrt(metrics.mean_squared_error(y_labeled, pred_train))
r2_5th_train = metrics.r2_score(y_labeled, pred_train)
print("第五轮模型训练的MAE：", mae_5th_train)
print("第五轮模型训练的RMSE：", rmse_5th_train)
print("第五轮模型训练的R2：", r2_5th_train)

fig, ax1 = plt.subplots(figsize=(20, 6))
plt.axis([n_initial - 20, n_sample + 20, -30, 60])
plt.xticks(size=18)
plt.yticks(size=18)
bwith = 3
ax1.spines["top"].set_linewidth(bwith)
ax1.spines["left"].set_linewidth(bwith)
ax1.spines["right"].set_linewidth(bwith)
ax1.spines["bottom"].set_linewidth(bwith)
ax1.tick_params(which='major', length=10, width=3)
plt.scatter(x_index_unlabeled, pred_5th, s=60, facecolors='none', edgecolors='grey', linewidths=3, label="Prediction")
plt.fill_between(x_index_unlabeled, pred_5th+std_5th, pred_5th-std_5th, facecolor='green', alpha=0.5, label="±1 Std Dev")
plt.legend(loc="best", fontsize=16)
plt.title('Prediction with Uncertainty-5th', fontsize=20)
ax1.get_xaxis().set_visible(True)
ax1.grid(False)
plt.tight_layout()
plt.savefig('energy-Al5-std.png', bbox_inches='tight', dpi=300)
plt.show()
fig, ax = plt.subplots(figsize=(5, 6))
plt.axis([-30, 90, -30, 90])
plt.xticks(size=18)
plt.yticks(size=18)
bwith=3
ax.spines["top"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)
ax.spines["bottom"].set_linewidth(bwith)
ax.tick_params(which='major',length=10,width=3)
ax.scatter(pred_train, y_labeled,  s=100, facecolors='none',  edgecolors='green',linewidths=3, label=("5th-Training"))
ax.scatter(pred_5th, y_unlabeled, color="red", s=100, label="Test")


line = mlines.Line2D([0, 1], [0, 1], color='black', linewidth=3)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel(r'$\Delta{G}$-predictions', size=18, weight="bold")
plt.ylabel(r'$\Delta{G}$-predictions',size=18, weight="bold")
plt.legend()
plt.legend(fontsize=18,loc="upper left",frameon=False)
plt.savefig('energy-Al5.png', bbox_inches='tight', dpi=300)
plt.show()

pred_5th=pd.DataFrame(pred_5th)
std_5th=pd.DataFrame(std_5th)
x_index_unlabeled=pd.DataFrame(x_index_unlabeled)
std_5th=np.hstack((x_index_unlabeled, std_5th, pred_5th))
std_5th=pd.DataFrame(std_5th)
std_5th = std_5th.rename(columns={
    std_5th.columns[0]: "query_indices", 
    std_5th.columns[1]: "std_5th",
    std_5th.columns[2]: "pred_5th"
    })

std_5th["std_5th"] = std_5th["std_5th"].astype(float)
std_5th["pred_5th"] = std_5th["pred_5th"].astype(float)

std_5th.to_csv('std_5th.csv', index=False)
top_mark_std_5th = std_5th.nlargest(mark, 'std_5th')
print("top_mark_std_5th:", top_mark_std_5th)
query_indices_std_5th = top_mark_std_5th['query_indices']
query_indices_std_5th = pd.DataFrame(query_indices_std_5th)
top_mark_pred_5th_min = std_5th.nsmallest(mark2, 'pred_5th')
print("top_mark_pred_5th_min:", top_mark_pred_5th_min)
query_indices_pred_5th_min = top_mark_pred_5th_min['query_indices']
query_indices_pred_5th_min = pd.DataFrame(query_indices_pred_5th_min)
while len(set(query_indices_std_5th['query_indices']).intersection(set(query_indices_pred_5th_min['query_indices']))) > 0:
    std_5th = std_5th[~std_5th['query_indices'].isin(query_indices_std_5th['query_indices'])]
    top_mark_pred_5th_min = std_5th.nsmallest(mark2, 'pred_5th')
    print("Re-finding top_mark_pred_5th_min:", top_mark_pred_5th_min)
    query_indices_pred_5th_min = top_mark_pred_5th_min['query_indices']
    query_indices_pred_5th_min = pd.DataFrame(query_indices_pred_5th_min)
top_mark_pred_5th_max = std_5th.nlargest(mark3, 'pred_5th')
print("top_mark_pred_5th_max:", top_mark_pred_5th_max)
query_indices_pred_5th_max = top_mark_pred_5th_max['query_indices']
query_indices_pred_5th_max = pd.DataFrame(query_indices_pred_5th_max)

while len(set(query_indices_std_5th['query_indices']).intersection(set(query_indices_pred_5th_max['query_indices']))) > 0:
    std_5th = std_5th[~std_5th['query_indices'].isin(query_indices_std_5th['query_indices'])]
    top_mark_pred_5th_max = std_5th.nlargest(mark3, 'pred_5th')
    print("Re-finding top_mark_pred_5th_max:", top_mark_pred_5th_max)
    query_indices_pred_5th_max = top_mark_pred_5th_max['query_indices']
    query_indices_pred_5th_max = pd.DataFrame(query_indices_pred_5th_max)

query_indices=pd.concat([query_indices_std_5th, query_indices_pred_5th_min,query_indices_pred_5th_max])
query_indices = query_indices.squeeze()
query_indices = query_indices.astype(int)
query_indices = np.array(query_indices)
print(query_indices)
print(query_indices.shape)
new_labeled_X = X.iloc[query_indices, :]
new_labeled_y = y[query_indices]
new_labeled_y = pd.DataFrame(new_labeled_y)
print("5th_new_labeled_y:", new_labeled_y)

labeled_indices = np.append(labeled_indices, query_indices)
X_labeled = X.iloc[labeled_indices, :]
y_labeled = y.iloc[labeled_indices]

unlabeled_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indices)
print("unlabeled_indices:", unlabeled_indices.shape)
X_unlabeled = X.iloc[unlabeled_indices, :]
y_unlabeled = y.iloc[unlabeled_indices]
x_index_unlabeled = x_index.iloc[unlabeled_indices]
x_index_info=x_complex[unlabeled_indices]

