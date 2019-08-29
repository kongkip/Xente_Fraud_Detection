import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

#%%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve
from collections import Counter

from sklearn.model_selection import KFold,StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler,OneHotEncoder
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve,classification_report, average_precision_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

# from mlxtend.evaluate import confusion_matrix
# from mlxtend.plotting import plot_confusion_matrix


import warnings
warnings.filterwarnings("ignore")

# load the dataset
df = pd.read_csv("data/training.csv")
test = pd.read_csv("data/test.csv")



# df.head()


df.columns
test.columns

train = df.drop("FraudResult", axis=1)
label = df["FraudResult"]
train.columns == test.columns

train.head()
test.head()

all_data = pd.concat([train, test],axis=0)

all_data.tail(2)



all_data["TransactionStartTime"] = pd.to_datetime(all_data.TransactionStartTime)

all_data.head(2)


# check null amounts
all_data.isnull().sum().max()

print("No Frauds", round(label.value_counts()[0]/len(df)*100,2), "% of the dataset")
print("Frauds", round(label.value_counts()[1]/len(df)*100,2), "% of the dataset")



colors = ["blue","red"]
sns.countplot("FraudResult", data=df, palette=colors)
plt.title("Class Distributions \n (0: No Fraud || 1: Fraud)", fontsize=14)
plt.show()


# Distribution of transaction
t_value = all_data["Value"].values
t_amount = all_data["Amount"].values

sns.distplot(t_value, color="r")
plt.title("Ditribution of Transaction Value", fontsize=14)
plt.xlim([min(t_value), max(t_value)])

sns.distplot(t_amount, color="b")
plt.title("Ditribution of Transaction Amount", fontsize=14)
plt.xlim([min(t_amount), max(t_amount)])
plt.show()



# Scale value and amount

# value_std_scaler = StandardScaler()
# amount_std_scaler = StandardScaler()
# all_data["scaled_value"] = value_std_scaler.fit_transform(all_data.Value.values.reshape(-1,1))
# all_data["scaled_amount"] = amount_std_scaler.fit_transform(all_data.Amount.values.reshape(-1,1))
# all_data.head()

dat


# all_data.drop(["Value", "Amount"], axis=1, inplace=True)
all_data.drop(["Amount"], axis=1, inplace=True)
all_data.head(1)


# remove id features
# all_data.columns
all_data.dtypes

col_to_remove = ['BatchId', 'AccountId', 'SubscriptionId',
                'CustomerId','ProviderId', 'ProductId'
                ,'ChannelId',"CurrencyCode","CountryCode"]
for i in col_to_remove:
    print(i)
    print(all_data[i].nunique())
col_to_remove = ["CurrencyCode","CountryCode"]

all_data.drop(col_to_remove, axis=1, inplace=True)

# create features from datetime
all_data["hour"] = all_data["TransactionStartTime"].dt.hour
all_data["day"] = all_data["TransactionStartTime"].dt.day
all_data["week"] = all_data["TransactionStartTime"].dt.week
all_data["month"] = all_data["TransactionStartTime"].dt.month
all_data["year"] = all_data["TransactionStartTime"].dt.year
all_data["minute"] = all_data["TransactionStartTime"].dt.minute

all_data.drop(['TransactionStartTime'], axis=1, inplace=True)
all_data.head()

# sns.countplot("CountryCode", data=df, palette=colors)
# plt.title("CountryCode Distribution")
# plt.show()

df["ProductCategory"].value_counts()
plt.figure(figsize=(12,6))
sns.countplot("ProductCategory", data=all_data)
plt.title("ProductCategory Distripution")
plt.show()

df["PricingStrategy"].value_counts()
plt.figure(figsize=(12,6))
sns.countplot("PricingStrategy", data=df)
plt.title("PricingStrategy Distripution")
plt.show()

# drop country code and CurrencyCode

# Convert ProductCategory to numerical
p_encoder = LabelEncoder()
cat_columns = ['BatchId', 'AccountId', 'SubscriptionId',
                'CustomerId','ProviderId', 'ProductId'
                ,'ChannelId',"ProductCategory"]
for i in cat_columns:
    all_data[i] = p_encoder.fit_transform(
                np.array(all_data[i]).reshape(-1, 1))
all_data.head(2)
# all_data["ProductCategory"] = p_encoder.fit_transform(np.array(
# all_data.ProductCategory).reshape(-1, 1))
# all_data.head()

train_data = all_data[:len(train)]
test_data = all_data[len(test):]

test_data.tail()

X = train_data.drop("TransactionId", axis=1)

# df["FraudResult"] = label
# Splitting the dataset
# y = df["FraudResult"]
#
#
# sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
#
# for train_index, test_index in sss.split(X,y):
#     print("Train:", train_index, "Test:", test_index)
#     original_Xtrain,original_Xest = X.iloc[train_index], X.iloc[test_index]
#     original_ytrain,original_ytest = y.iloc[train_index], y.iloc[test_index]
#
# # Turn into array
#
# original_Xtrain = original_Xtrain.values
# original_Xest = original_Xest.values
# original_ytrain = original_ytrain.values
# original_ytest = original_ytest.values
#
# train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
# test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
# print("-"*100)
#
# print("LabeL Distributions: \n")
# print(train_counts_label/ len(original_ytrain))
# print(test_counts_label/ len(original_ytest))
#
# df = df.sample(frac=1)
#
# len(df.loc[df["FraudResult"] == 1])
#
# fraud_df = df.loc[df["FraudResult"] == 1]
# non_fraud_df = df.loc[df["FraudResult"]==0][:193]
#
#
# normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
#
# new_df = normal_distributed_df.sample(frac=1,random_state=42)
#
# new_df.head()
#
#
# print("Distribution of FraudResult in the sample dataset")
# print(new_df["FraudResult"].value_counts()/len(new_df))
#
# sns.countplot("FraudResult", data=new_df, palette=colors)
# plt.title("Equally distributed Classes")
# plt.show()
#
# f, (ax1,ax2) = plt.subplots(2,1, figsize=(24,20))
#
# # Entire Dataframe
# corr = df.corr()
# sns.heatmap(corr, cmap="coolwarm_r", annot_kws = {"size":20}, ax=ax1)
# ax1.set_title("Imbalanced Correlation Matrix",fontsize=14)
#
# sub_sample_corr = new_df.corr()
# sns.heatmap(sub_sample_corr, cmap="coolwarm_r", annot_kws = {"size":20}, ax=ax2)
# ax2.set_title("Subsample Correlation Matrix", fontsize=14)
# plt.show()
#
# # ProductCategory, scaled_value, scaled_amount, hour and year have positive Correlation
# # PricingStrategy, day,week and month have negative Correlation
#
# # remove scalled_amount, stroongly positively relted to scaled value
# #new_df.drop(['scaled_amount'], axis=1, inplace=True)
# # Box plots
# f, axes = plt.subplots(ncols=4, figsize=(20,4))
#
# sns.boxplot(x="FraudResult", y="ProductCategory", data=new_df, palette=colors, ax=axes[0])
# axes[0].set_title("ProductCategory vs Fraud Positive Correlation")
#
# sns.boxplot(x="FraudResult", y="scaled_value", data=new_df, palette=colors, ax=axes[1])
# axes[1].set_title("scaled_value vs Fraud Positive Correlation")
#
# # sns.boxplot(x="FraudResult", y="scaled_amount", data=new_df, palette=colors, ax=axes[2])
# # axes[2].set_title("scaled_amount vs Fraud Positive Correlation")
#
# sns.boxplot(x="FraudResult", y="hour", data=new_df, palette=colors, ax=axes[3])
# axes[3].set_title("hour vs Fraud Positive Correlation")
#
# plt.show()
#
#
# f, axes = plt.subplots(ncols=4, figsize=(20,4))
#
# sns.boxplot(x="FraudResult", y="PricingStrategy", data=new_df, palette=colors, ax=axes[0])
# axes[0].set_title("PricingStrategy vs Fraud negative Correlation")
#
# sns.boxplot(x="FraudResult", y="day", data=new_df, palette=colors, ax=axes[1])
# axes[1].set_title("day vs Fraud negative Correlation")
#
# sns.boxplot(x="FraudResult", y="week", data=new_df, palette=colors, ax=axes[2])
# axes[2].set_title("week vs Fraud negative Correlation")
#
# sns.boxplot(x="FraudResult", y="month", data=new_df, palette=colors, ax=axes[3])
# axes[3].set_title("month vs Fraud negative Correlation")


# Anomaly detection
# f, (ax1,ax2) = plt.subplots(1,2, figsize=(20,4))
#
#
# scaled_value_dist  = new_df['scaled_value'].loc[new_df["FraudResult"]==1].values
# sns.distplot(scaled_value_dist,ax=ax1, fit=norm, color="red")
# ax1.set_title("scaled_value Distribution \n (Fraud Transactions)")
#
# scaled_amount_dist  = new_df['scaled_amount'].loc[new_df["FraudResult"]==1].values
# sns.distplot(scaled_amount_dist,ax=ax2, fit=norm, color="blue")
# ax2.set_title("scaled_amount Distribution \n (Fraud Transactions)")
#
# plt.show()
#
#
# scaled_value_fraud = new_df['scaled_value'].loc[new_df["FraudResult"]==1].values
# q25, q75 = np.percentile(scaled_value_fraud, 25), np.percentile(scaled_value_fraud, 75)
# print("Quartile 25: {} | Quartile 75: {}".format(q25,q75))
# scaled_value_1qr = q75 - q25
# print("iqr: {}".format(scaled_value_1qr))
#
# scaled_value_cut_off = scaled_value_1qr*1.5
# scaled_value_lower, scaled_value_upper = q25-scaled_value_cut_off, q75-scaled_value_cut_off
#
# outliers = [x for x in scaled_value_fraud if x < scaled_value_lower or x > scaled_value_upper]
# print("Feature Scaled value for fraud cases: {}".format(len(outliers)))
# print("Scaled Value outliers:{}".format(outliers))
#
# new_df = new_df.drop(new_df[(new_df["scaled_value"] > scaled_value_upper) | (new_df["scaled_value"] < scaled_value_upper)].index)
# print("==="*45)

# new_df.head(1)
# # clasifiers under_sampling
# X = new_df.drop("FraudResult", axis=1).values
# y = new_df["FraudResult"].values

#
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#
# classifiers = {
#         "LogisticRegression":LogisticRegression(),
#         "KNeighborsClassifier": KNeighborsClassifier(),
#         "Support Vector Classifier": SVC(),
#         "DecisionTreeClassifier": DecisionTreeClassifier()
#
#         }
#
# for key, classifier in classifiers.items():
#     classifier.fit(X_train, y_train)
#     train_score = cross_val_score(classifier, X_train, y_train)
#     test_score = cross_val_score(classifier, X_test, y_test)
#     print("Classifiers: ", classifier.__class__.__name__, "Has Training Score of", round(train_score.mean()))
#     print("Classifiers: ", classifier.__class__.__name__, "Has Testing Score of", round(test_score.mean()))
#
#
# # Prepare the test set
# sub = pd.read_csv("submission/ample_submission.csv")
# # test_set.head()
#
# # sub.head()
#
# # test_set.columns
# id = test_data["TransactionId"]
# test_data = test_data.drop(["TransactionId"],axis=1)
# test_data.head(2)



# log_reg = LogisticRegression()
# log_reg.fit(X, y)
#
# y_pred = log_reg.predict(X)
# print(classification_report(y, y_pred))
#
# pred = log_reg.predict(test_data)
#
# submission = pd.DataFrame(data=id,columns=["TransactionId"])
# submission["FraudResult"] = pred
#
#
#
# submission.head(2)
#
# submission.to_csv("initial_submit.csv",index=False)


# over_sampling with SMOTE techinique
cat_features = cat_columns
cat_features.append("PricingStrategy")
cat_features

cat = CatBoostClassifier(cat_features=cat_features)


cat.fit(X,y)

y_pred = cat.predict(X)
print(classification_report(y, y_pred))
id = test_data["TransactionId"]
test_data = test_data.drop(["TransactionId"],axis=1)
test_data.head(2)

pred = cat.predict(test_data)
#
submission = pd.DataFrame(data=id,columns=["TransactionId"])
submission["FraudResult"] = pred



submission.head(2)
submission.tail(2)

submission.to_csv("submission/cat_features_submit.csv",index=False)
