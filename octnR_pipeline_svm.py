#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[66]:


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


import numpy as np

# Load Dataset
data = pd.read_excel('/Users/nangosyah/Desktop/MSc. Statistics & Data Science/Thesis ML:AI/Thesis Final/new models/rright.xlsx')

# Replace 'Not Severe' and 'Severe' with 0 and 1
data['EDSS_Binary'] = data['EDSS_Binary'].replace({'Not Severe': 0, 'Severe': 1}).astype(int)

# Replace 'inf' and '-inf' with 'NaN'
data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[68]:


# separate features and target
X = data.drop(columns=['EDSS_Binary', 'PatientID'])
y = data['EDSS_Binary']


# In[69]:


# Preprocessing
# categorical and numerical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(include=['float64', 'int64']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


# In[ ]:


# split data into training and testing sets
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, stratify=y, random_state=42)

print("\nTraining data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts())

print("\nClass distribution in test set:")
print(pd.Series(y_test).value_counts())


# In[ ]:


from imblearn.over_sampling import SMOTENC
import pandas as pd

# class Imbalance
smote_nc = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in cat_cols], random_state=42)
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

# column names
num_features = num_cols.tolist()
cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)

# combine numerical and categorical features
all_features = num_features + list(cat_features)
X_resampled_df = pd.DataFrame(X_train_resampled, columns=all_features)

print("\nClass distribution after SMOTE resampling:")
print(pd.Series(y_train_resampled).value_counts())

print("Proportion of the Minority Class in train set:" + str(round(y_train.sum()/len(y_train)*100,2)) + "%")
print("Proportion of the Minority Class in test set:"+ str(round(y_test.sum()/len(y_test)*100,2)) + "%")


# In[ ]:


from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(42)

# ElasticNet
lasso = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=5, max_iter=10000, random_state=42)

# train the model on the resampled data
lasso.fit(X_resampled_df, y_train_resampled)

# selected features
feature_mask = lasso.coef_ != 0

# feature names
num_features = num_cols.tolist()
cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(cat_cols)
all_features = num_features + cat_features.tolist()
selected_features = np.array(all_features)[feature_mask.flatten()]
print("Lasso selected features:", selected_features)

X_train_df = pd.DataFrame(X_train, columns=all_features)
X_test_df = pd.DataFrame(X_test, columns=all_features)

X_train_selected = X_train_df[selected_features]
X_test_selected = X_test_df[selected_features]

print("\nShape of data with selected features (training):", X_train_selected.shape)
print("\nShape of data with selected features (test):", X_test_selected.shape)


# In[74]:


# correct number of rows
y_train_resampled = y_train_resampled[:len(X_train_selected)]


# In[ ]:


from sklearn.svm import SVC

# train the SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_selected, y_train_resampled)


# In[76]:


# predict and evaluate the model
y_pred_svm = svm_model.predict(X_test_selected)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# classification report and confusion matrix
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))


# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve
from plotnine import ggplot, aes, geom_line, labs, theme_bw

# predict probabilities
y_prob = svm_model.predict_proba(X_test_selected)[:, 1]

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# calculate AUC
auc = roc_auc_score(y_test, y_prob)

roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})

# plot ROC curve
roc_plot = (ggplot(roc_data, aes(x='False Positive Rate', y='True Positive Rate'))
            + geom_line()
            + labs(title=f'ROC Curve (AUC = {auc:.2f})')
            + theme_bw())

roc_plot.show()


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np

# Hyperparameter grid for SVM
svm_param_dist = {
    'C': [0.01, 0.1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
    'class_weight': ['balanced', None]
}


# RandomizedSearchCV
svm_random_search = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_param_dist, cv=5, n_iter=10, random_state=42, n_jobs=-1)
svm_random_search.fit(X_train_selected, y_train_resampled)

# best parameters
print("Best parameters for SVM:", svm_random_search.best_params_)

# cross-validation
cv_scores_svm = cross_val_score(svm_random_search.best_estimator_, X_train_selected, y_train_resampled, cv=10)
print("Cross-validation score for SVM:", cv_scores_svm.mean())


# In[ ]:


# best SVM model
best_svm_model = svm_random_search.best_estimator_

# fit the best model to the entire training data
best_svm_model.fit(X_train_selected, y_train_resampled)

# predictions and evaluate on test data
y_pred_svm = best_svm_model.predict(X_test_selected)

# classification report and confusion matrix
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))


# In[ ]:


from imblearn.metrics import classification_report_imbalanced

# classification report
print(classification_report_imbalanced(y_test, y_pred_svm))


# In[ ]:


# Import necessary libraries
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred_svm)
print("\nBalanced Accuracy:", round(balanced_acc, 4))

# Calculate metrics for the "severe" class
target_names = {0: 'Not Severe', 1: 'Severe'}
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_svm, labels=[1], average=None)

print("\nMetrics for Severe Class (1):")
print(f"Precision: {precision[0]:.4f}")
print(f"Recall: {recall[0]:.4f}")
print(f"F1 Score: {f1_score[0]:.4f}")


# In[ ]:


from sklearn.inspection import permutation_importance

# permutation importance
perm_importance = permutation_importance(
    best_svm_model,
    X_test_selected, 
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

feature_importance_df = pd.DataFrame({
    'Feature': X_test_selected.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)


# In[ ]:


import matplotlib.pyplot as plt

# plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df['Feature'], 
    feature_importance_df['Importance'], 
    color='skyblue')
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# predicted probabilities for each model
y_pred_prob_svm = best_svm_model.decision_function(X_test_selected)

# compute ROC curve and AUC for each model
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})", color="orange")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")

# customize plot
plt.title("ROC Support Vector Machine - Right Eye", fontsize=20, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

# customize legend
plt.legend(loc="lower right", fontsize=14, title="Legend", title_fontsize=16, frameon=True)
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()

