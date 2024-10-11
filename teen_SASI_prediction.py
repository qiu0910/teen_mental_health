# significance testing using Chi-square test and t-test
# using random forest for prediction
# stratified nested cross-validation procedure for hyperparameter tuning, model training, and unbiased estimation
# Shapley Additive exPlanations (SHAP) method for in-depth interpretation


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, norm
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.formula.api import logit
from statsmodels.stats.multicomp import MultiComparison
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# sampling
from imblearn.over_sampling import SMOTE
from collections import Counter

# model
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.svm import SVC

# evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report, \
    accuracy_score, recall_score
from sklearn import metrics

# SHAP method
import shap

# cross validation
from sklearn.model_selection import KFold, StratifiedKFold

# feature selection
from minepy import MINE
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
import mifs

# data
teen_sasi = pd.read_csv('your_folder/teen_SASI.csv')

# data processing, do your data cleaning and data encoding here

# remove duplicate samples
print('With' if any(teen_sasi.duplicated()) else 'Without', 'duplicate samples')
teen_sasi = teen_sasi.drop_duplicates()

# missing values
print('With' if any(teen_sasi.isnull()) else 'Without', 'missing values')

# store new data
teen_sasi.to_csv('your_folder/teen_sasi_after_processing.csv')

# See if the samples are balanced
sns.set_style('whitegrid')
sns.countplot(x='SASI', data=teen_sasi, palette='RdBu_r')
plt.savefig('your_folder/balance_ornot.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(1)
plt.close()

# significance testing
sasi_0 = teen_sasi[teen_sasi['SASI'] == 0]
sasi_1 = teen_sasi[teen_sasi['SASI'] == 1]
with open('your_folder/Chi_square_Ttest_results.txt', 'w') as f1:
    print("Significance Testing", file=f1)
    # Chi-square test
    # for binary variables
    for col in ['your_features']:
        print(f"{col}:", file=f1)
        value_counts = teen_sasi[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(teen_sasi) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)
        value_counts = sasi_1[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(sasi_1) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)
        value_counts = sasi_0[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(sasi_0) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)

        contingency_table = pd.crosstab(teen_sasi[col], teen_sasi['SASI'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f'{col} and SASI, the chi-square test result is: χ^2 = {chi2:.2f} ({dof}), p = {p_value:.3f}',file = f1)

        table = Table2x2(contingency_table)
        odds_ratio = table.oddsratio
        odds_ratio_ci = table.oddsratio_confint()
        print(f"Odds Ratio and 95% confidence interval: {odds_ratio:.2f} ({odds_ratio_ci[0]:.2f}-{odds_ratio_ci[1]:.2f})", file=f1)

    # for non-binary variables
    for col in ['your_features']:
        print(f"{col}:", file=f1)
        value_counts = teen_sasi[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(teen_sasi) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)
        value_counts = sasi_1[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(sasi_1) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)
        value_counts = sasi_0[col].value_counts()
        value_c = sorted(value_counts.items(), key=lambda x: int(x[0]))
        for value, count in value_c:
            percentage = count / len(sasi_0) * 100
            print(f"{int(count)} ({percentage:.1f}%)", file=f1)

        contingency_table = pd.crosstab(teen_sasi[col], teen_sasi['SASI'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f'{col} and SASI, the chi-square test result is: χ^2 = {chi2:.2f} ({dof}), p = {p_value:.3f}', file=f1)

        for i in range(contingency_table.shape[0]):
            if i == 0 :
                continue
            a = contingency_table.iloc[i, 1]
            b = contingency_table.iloc[i, 0]
            c = contingency_table.iloc[1, 1]
            d = contingency_table.iloc[1, 0]

            odds_ratio = (a * d) / (b * c)
            se_log_or = ((1 / a) + (1 / b) + (1 / c) + (1 / d)) ** 0.5

            alpha = 0.05
            z = norm.ppf(1 - alpha / 2)

            lower_ci = np.exp(np.log(odds_ratio) - z * se_log_or)
            upper_ci = np.exp(np.log(odds_ratio) + z * se_log_or)

            print(f"Odds Ratio for your_feature level {i + 1}: {odds_ratio:.2f} ({lower_ci:.2f}-{upper_ci:.2f})",file=f1)

    # t - test
    # for continuous variables
    for col in ['your_features']:
        print(f"{col}:", file=f1)
        mean = teen_sasi[col].mean()
        std = teen_sasi[col].std()
        print(f"{mean:.2f} ± {std:.2f}",file=f1)
        mean = sasi_1[col].mean()
        std = sasi_1[col].std()
        print(f"{mean:.2f} ± {std:.2f}", file=f1)
        mean = sasi_0[col].mean()
        std = sasi_0[col].std()
        print(f"{mean:.2f} ± {std:.2f}", file=f1)
        t_stat, p_value = ttest_ind(teen_sasi.loc[teen_sasi['SASI']==1, col], teen_sasi.loc[teen_sasi['SASI']==0, col])
        print(f'{col} and SASI, the t-test result is: t = {t_stat:.2f}, p = {p_value:.3f}',file = f1)


# check the data
print(teen_sasi.head())
teen_sasi.info()
with open('your_folders/teen_sasi_describe.txt', 'w') as f2:
    print(teen_sasi.describe().T, file=f2)

fig = plt.figure(figsize=(20, 65))
sns.set_theme(style='dark')
# get features
teen_sasi2 = teen_sasi
feature_x = teen_sasi2.drop(['SASI'], axis=1)
feature_drop = list(feature_x.columns)
countnum = 0
for n in feature_drop:
    countnum = countnum + 1
    plt.subplot(21, 3, countnum)
    sns.countplot(data=teen_sasi, x=n, hue='SASI')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
                    wspace=None, hspace=0.45)
plt.savefig('your_folders/SASI_data_distribution.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(1)
plt.close()

# heatmap for correlation
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
pearson_mat = teen_sasi.corr(method='spearman')
plt.figure(figsize=(50, 50))
ax = sns.heatmap(pearson_mat, square=True, annot=True, cmap='YlGnBu')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xticks(rotation=-30)
plt.yticks(rotation=-30)
plt.savefig('your_folders/heatmap.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(1)
plt.close()

# features and target variable
total_x= teen_sasi.drop(['SASI'], axis=1)
total_y = teen_sasi['SASI']

feature = list(total_x.columns)

# split data
train_x, test_X, train_y, test_Y = train_test_split(total_x, total_y, test_size=0.3, random_state=0, stratify=total_y)
# store original test data
new_test = pd.concat([test_X, test_Y], axis=1)

# data processing after sampling
new_train = pd.concat([train_x, train_y], axis=1)
# remove duplicate samples
print('With' if any(new_train.duplicated()) else 'Without', 'duplicate samples')
new_train = new_train.drop_duplicates()

# stratified nested cross-validation
# store the outputs
with open('your_folders/RF-train_Result-tenfold.txt', 'w') as f3:
    print("\nFinally results of RF——10fold:", file=f3)
with open('your_folders/RF-train_best_para.txt', 'w') as f4:
    print("\nBest parameter in each split:", file=f4)

n_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# hyper-parameter settings
n_estimators = [100, 200, 300, 500, 700, 1000]
max_features = ["sqrt", "log2"]
max_depth = np.arange(10, 50, step=10)

final_importance = np.zeros(shape=(train_x.shape[1]))
best_score = 0
best_model = RandomForestClassifier()
kfold_count=0
for train_index, test_index in n_fold.split(train_x, train_y):
    kfold_count = kfold_count+1
    this_train_x, this_train_y = train_x.iloc[train_index], train_y.iloc[train_index]  # training set in fold
    this_test_x, this_test_y = train_x.iloc[test_index], train_y.iloc[test_index]      # validation set in fold

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth
    }

    random_rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_fold, scoring="roc_auc", n_jobs=-1)
    random_rf_cv.fit(this_train_x, this_train_y)
    best_params_ = random_rf_cv.best_params_
    best_score_ = random_rf_cv.best_score_

    with open('your_folders/RF-train_best_para.txt', 'a') as f4:
        print("best_params and best_score(roc_auc) of Random Forest——split{}:".format(kfold_count), file=f4)
        print(best_params_,file=f4)
        print(best_score_, file=f4)

    RF_model = random_rf_cv.best_estimator_
    RF_pre = RF_model.predict(this_test_x)
    MC = metrics.confusion_matrix(this_test_y, RF_pre)
    with open('your_folders/RF-train_Result-tenfold.txt', 'a') as f3:
        print("\nFinally results of RF——splits{}:".format(kfold_count), file=f3)
        print("Accuracy of Random Forest on training set:  {:.3f}".format(RF_model.score(this_train_x, this_train_y)), file=f3)
        print("Accuracy of Random Forest on testing set:  {:.3f}\n".format(RF_model.score(this_test_x, this_test_y)), file=f3)
        print(classification_report(this_test_y, RF_pre), file=f3)
        print("confusion_matrix of Random Forest:\n{}".format(MC), file=f3)
        plt.figure(figsize=(30, 30))
        plt.matshow(MC, cmap=plt.cm.GnBu)
        iters = np.reshape([[[i, j] for j in range(len(MC))] for i in range(len(MC))], (MC.size, 2))
        for i, j in iters:
            if (i == j):
                plt.text(j, i, format(MC[i, j]), va='center', ha='center', fontsize=12, color='white',
                         weight=5)
            else:
                plt.text(j, i, format(MC[i, j]), va='center', ha='center', fontsize=12)
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig('your_folders/train_confusion matrix-split{}.png'.format(kfold_count), bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    # feature importance of RF
    importance2 = RF_model.feature_importances_
    final_importance = final_importance + importance2
    feat_labels = this_train_x.columns[0:]
    indices = np.argsort(importance2)[::-1]
    with open('your_folders/RF-train_Result-tenfold.txt', 'a') as f3:
        print("Feature importance of Random Forest:\n", file=f3)
        for f in range(train_x.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance2[indices[f]]), file=f3)
    final_im = pd.Series(RF_model.feature_importances_, index=feat_labels)
    final_im = final_im.sort_values()
    ax = final_im.plot.barh(figsize = (10, 6.18), title="Feature Importance by RandomForest, split{}".format(kfold_count))
    plt.savefig('your_folders/RF-train_importance-ranking-split{}.png'.format(kfold_count), bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # ROC curve
    y_pred_rf = RF_model.predict_proba(this_test_x)[:, 1]
    fpr_grd_lm, tpr_grd_lm, threshold = roc_curve(this_test_y, y_pred_rf)
    roc_auc = auc(fpr_grd_lm, tpr_grd_lm)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr_grd_lm, tpr_grd_lm, color='darkorange',lw=2, label='ROC curve (area = %0.2f)'%roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Area Under the Receiver Operating Characteristic Curve')
    plt.savefig('your_folders/RF-train_ROC_curve-split{}.png'.format(kfold_count), bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    if (best_score < roc_auc):
        best_score = roc_auc
        best_model = RF_model

# average importance scores after cross-evaluation
final_importance = final_importance/10
indices = np.argsort(final_importance)[::-1]
feat_labels = train_x.columns[0:]
with open('your_folders/RF-train_final_importance-tenfold.txt', 'w') as f5:
    for f in range(train_x.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], final_importance[indices[f]]), file=f5)
final_rank = pd.Series(final_importance, index=feat_labels)
final_rank = final_rank.sort_values()
ax = final_rank.plot.barh(figsize = (10, 6.18), title="Feature Importance by RandomForest")
plt.savefig('your_folders/RF-train_final-ranking.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(1)
plt.close()

best_para={"n_estimators": best_model.n_estimators,"max_features": best_model.max_features,"max_depth": best_model.max_depth,"min_samples_split": best_model.min_samples_split,"min_samples_leaf": best_model.min_samples_leaf}

RF_pre = best_model.predict(test_X)
MC = metrics.confusion_matrix(test_Y, RF_pre)
with open('your_folders/RF-final_Result-tenfold.txt', 'w') as f6:
    print("\nFinal results of RF", file=f6)
    print("Accuracy of Random Forest on training set:  {:.3f}".format(best_model.score(train_x, train_y)),file=f6)
    print("Accuracy of Random Forest on testing set:  {:.3f}\n".format(best_model.score(test_X, test_Y)),file=f6)
    print(classification_report(test_Y, RF_pre), file=f6)
    print("confusion_matrix of Random Forest:\n{}".format(MC), file=f6)
    print("Feature importance of Random Forest:\n", file=f6)
    plt.figure(figsize=(30, 30))
    plt.matshow(MC, cmap=plt.cm.GnBu)
    iters = np.reshape([[[i, j] for j in range(len(MC))] for i in range(len(MC))], (MC.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i, format(MC[i, j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5)
        else:
            plt.text(j, i, format(MC[i, j]), va='center', ha='center', fontsize=12)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('your_folders/final_confusion matrix.png', bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
importance2 = best_model.feature_importances_
feat_labels = train_x.columns[0:]
indices = np.argsort(importance2)[::-1]
with open('your_folders/RF-final_Result-tenfold.txt', 'a') as f6:
    for f in range(train_x.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance2[indices[f]]), file=f6)
final_im = pd.Series(best_model.feature_importances_, index=feat_labels)
final_im = final_im.sort_values()
ax = final_im.plot.barh(figsize=(10, 6.18), title="Final Feature Importance by RandomForest")
plt.savefig('your_folders/RF-final_importance-ranking.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(1)
plt.close()

# ROC curve
y_pred_rf = best_model.predict_proba(test_X)[:, 1]
fpr_grd_lm, tpr_grd_lm, threshold = roc_curve(test_Y, y_pred_rf)
roc_auc = auc(fpr_grd_lm, tpr_grd_lm)
plt.figure(figsize=(8, 5))
plt.plot(fpr_grd_lm, tpr_grd_lm, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under the Receiver Operating Characteristic Curve')
plt.savefig('your_folders/RF-final_ROC_curve.png',bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
plt.close()


## SHAP method
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(test_X)
svalue = pd.DataFrame(shap_values[1])
svalue.to_csv('your_folders/shap_values.csv')

# summary_plot

shap.summary_plot(shap_values[1], test_X,show = False,max_display=20)
plt.savefig('your_folders/shap_summary_plot2.png',bbox_inches='tight')
plt.close()

# feature importance
shap.summary_plot(shap_values[1], test_X,plot_type="bar",show = False,max_display=20)
plt.savefig('your_folders/shap_summary_barplot.png',bbox_inches='tight')
plt.close()

# single sample
Pre_sasi = np.where(RF_pre==1)
Pre_sasi2 = np.where(RF_pre==0)

new_test=new_test.reset_index(drop=False)
a = new_test[(new_test.SASI==1)&(new_test.Sex==0)&(new_test.Only_child==1)].index
same_boy_sasi = np.intersect1d(Pre_sasi,a)
# sample_boy_index = min(same_boy_sasi)
with open('your_folders/sasi_onlyboy.txt', 'w') as f7:
    for sbi in same_boy_sasi:
        shap.force_plot(explainer.expected_value[1], shap_values[1][sbi,:], test_X.iloc[sbi,:],show = False,matplotlib=True,text_rotation=-12)
        print("{}\n".format(test_X.iloc[sbi,:]), file=f7)
        plt.savefig('your_folders/sasi_onlyboy_{}.png'.format(sbi),bbox_inches='tight')
        plt.close()

b = new_test[(new_test.SASI==1)&(new_test.Sex==1)&(new_test.Only_child==1)].index
same_girl_sasi = np.intersect1d(Pre_sasi,b)
with open('your_folders/sasi_onlygirl.txt', 'w') as f8:
    for sgi in same_girl_sasi:
        shap.force_plot(explainer.expected_value[1], shap_values[1][sgi,:], test_X.iloc[sgi,:],show = False,matplotlib=True,text_rotation=-12)
        print("{}\n".format(test_X.iloc[sgi, :]), file=f8)
        plt.savefig('your_folders/sasi_onlygirl_{}.png'.format(sgi),bbox_inches='tight')
        plt.close()

c = new_test[(new_test.SASI==0)&(new_test.Sex==0)&(new_test.Only_child==1)].index
same_boy_nosasi = np.intersect1d(Pre_sasi2,c)
with open('your_folders/nosasi_onlyboy.txt', 'w') as f9:
    for nbi in same_boy_nosasi:
        shap.force_plot(explainer.expected_value[1], shap_values[1][nbi,:], test_X.iloc[nbi,:],show = False,matplotlib=True,text_rotation=-12)
        print("{}\n".format(test_X.iloc[nbi, :]), file=f9)
        plt.savefig('your_folders/nosasi_onlyboy2_{}.png'.format(nbi),bbox_inches='tight')
        plt.close()

d = new_test[(new_test.SASI==0)&(new_test.Sex==1)&(new_test.Only_child==1)].index
same_girl_nosasi = np.intersect1d(Pre_sasi2,d)
with open('your_folders/nosasi_onlygirl.txt', 'w') as f10:
    for ngi in same_girl_nosasi:
        shap.force_plot(explainer.expected_value[1], shap_values[1][ngi,:], test_X.iloc[ngi,:],show = False,matplotlib=True,text_rotation=-12)
        print("{}\n".format(test_X.iloc[ngi, :]), file=f10)
        plt.savefig('your_folders/nosasi_onlygirl2_{}.png'.format(ngi),bbox_inches='tight')
        plt.close()

# dependence
shap.dependence_plot('Borderline_personality ',shap_values,test_X,show= False,interaction_index='Depression')
plt.savefig('your_folders/shap_dependence_Borderline_personality +Depressed.png',bbox_inches='tight')
plt.close()

shap.dependence_plot('Borderline_personality ',shap_values,test_X,show= False,interaction_index='Despair')
plt.savefig('your_folders/shap_dependence_Borderline_personality +Despair.png',bbox_inches='tight')
plt.close()

shap.dependence_plot('Cognitive_reappraisal',shap_values,test_X,show= False,interaction_index='Depression')
plt.savefig('your_folders/shap_dependence_Cognitive_reappraisal+Depressed.png',bbox_inches='tight')
plt.close()

shap.dependence_plot('Cognitive_reappraisal',shap_values,test_X,show= False,interaction_index='Despair')
plt.savefig('your_folders/shap_dependence_Cognitive_reappraisal+Despair.png',bbox_inches='tight')
plt.close()
