#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[51]:


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


# In[53]:


# separate features and target
X = data.drop(columns=['EDSS_Binary', 'PatientID'])
y = data['EDSS_Binary']


# In[54]:


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


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

np.random.seed(42)

# parameter grid
param_grid = {
    'penalty': ['elasticnet'],
    'C': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'solver': ['saga'],
    'max_iter': [10000]
}

# logistic regression
logreg = LogisticRegression(random_state=42)

# perform GridSearchCV
grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# train using the resampled training data
grid_search.fit(X_resampled_df, y_train_resampled)

# best parameters
best_params = grid_search.best_params_
print("Best parameters found:", best_params)

# train the final model with the best parameters
best_model = grid_search.best_estimator_

# selected features
feature_mask = best_model.coef_ != 0

# feature names
num_features = num_cols.tolist()
cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(cat_cols)

# combine the feature names
all_features = num_features + cat_features.tolist()
selected_features = np.array(all_features)[feature_mask.flatten()]
print("Selected features:", selected_features)

X_train_df = pd.DataFrame(X_train, columns=all_features)
X_test_df = pd.DataFrame(X_test, columns=all_features)

X_train_selected = X_train_df[selected_features]
X_test_selected = X_test_df[selected_features]

print("\nShape of data with selected features (training):", X_train_selected.shape)
print("\nShape of data with selected features (test):", X_test_selected.shape)


# In[59]:


# correct number of rows
y_train_resampled = y_train_resampled[:len(X_train_selected)]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# model building and evaluation
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_selected, y_train_resampled)


# In[61]:


# predict and evaluate the model
y_pred_knn = knn_model.predict(X_test_selected)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve
from plotnine import ggplot, aes, geom_line, labs, theme_bw


# predict probabilities
y_prob_knn = knn_model.predict_proba(X_test_selected)[:, 1]

# calculate ROC curve
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_prob_knn)

# calculate AUC
auc_knn = roc_auc_score(y_test, y_prob_knn)

roc_data_knn = pd.DataFrame({
    'False Positive Rate': fpr_knn,
    'True Positive Rate': tpr_knn,
    'Thresholds': thresholds_knn
})

# plot ROC curve
roc_plot = (ggplot(roc_data_knn, aes(x='False Positive Rate', y='True Positive Rate'))
            + geom_line()
            + labs(title=f'ROC Curve (AUC = {auc_knn:.2f})')
            + theme_bw())

roc_plot.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np

# Hyperparameter grid
knn_param_dist = {
    'n_neighbors': np.arange(3, 21, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# RandomizedSearchCV
knn_random_search = RandomizedSearchCV(KNeighborsClassifier(), knn_param_dist, cv=5, n_iter=10, random_state=42, n_jobs=-1)
knn_random_search.fit(X_train_selected, y_train)

# best parameters
print("Best parameters for KNN:", knn_random_search.best_params_)

# cross-validation
cv_scores_knn = cross_val_score(knn_random_search.best_estimator_, X_train_selected, y_train, cv=10)
print("Cross-validation score for KNN:", cv_scores_knn.mean())


# In[ ]:


# best KNN model
best_knn_model = knn_random_search.best_estimator_

# fit the best model to the entire training data
best_knn_model.fit(X_train_selected, y_train_resampled)

# best model
y_pred_knn = best_knn_model.predict(X_test_selected)

# classification report and confusion matrix
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# In[ ]:


from imblearn.metrics import classification_report_imbalanced

# classification report
print(classification_report_imbalanced(y_test, y_pred_knn))


# In[ ]:


# Import necessary libraries
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred_knn)
print("\nBalanced Accuracy:", round(balanced_acc, 4))

# calculate metrics for the "severe" class
target_names = {0: 'Not Severe', 1: 'Severe'}
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_knn, labels=[1], average=None)

print("\nMetrics for Severe Class (1):")
print(f"Precision: {precision[0]:.4f}")
print(f"Recall: {recall[0]:.4f}")
print(f"F1 Score: {f1_score[0]:.4f}")


# In[ ]:


from sklearn.inspection import permutation_importance
import pandas as pd

# compute permutation importance
perm_importance = permutation_importance(best_knn_model, X_test_selected, y_test, n_repeats=10, random_state=42)

feature_importance_df = pd.DataFrame({
    'Feature': X_test_selected.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# feature importance
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
plt.title('SVM Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# predicted probabilities
y_pred_prob_knn = best_knn_model.predict_proba(X_test_selected)[:, 1]

# compute ROC curve and AUC
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.2f})", color="red")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")

# customize plot
plt.title("ROC KNN - Right Eye")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# predicted probabilities
y_pred_prob_knn = best_knn_model.predict_proba(X_test_selected)[:, 1]

# compute ROC curve and AUC
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# plot ROC curves
plt.figure(figsize=(12, 9))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.2f})", color="red", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Chance", linewidth=2)

# customize plot
plt.title("ROC Curve for KNN - Right Eye", fontsize=20, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

# customize legend
plt.legend(loc="lower right", fontsize=14, title="Legend", title_fontsize=16, frameon=True)
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()

