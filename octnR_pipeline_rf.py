#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[50]:


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


# In[52]:


# separate features and target
X = data.drop(columns=['EDSS_Binary', 'PatientID'])
y = data['EDSS_Binary']


# In[53]:


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


# In[58]:


# correct number of rows
y_train_resampled = y_train_resampled[:len(X_train_selected)]


# In[ ]:


# model building and evaluation
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_selected, y_train_resampled)


# In[60]:


# predict and evaluate the model
y_pred = rf_model.predict(X_test_selected)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve
from plotnine import ggplot, aes, geom_line, labs, theme_bw


# Predict probabilities
y_prob_rf = rf_model.predict_proba(X_test_selected)[:, 1]

# Calculate ROC curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)

# AUC
auc_rf = roc_auc_score(y_test, y_prob_rf)

roc_data_rf = pd.DataFrame({
    'False Positive Rate': fpr_rf,
    'True Positive Rate': tpr_rf,
    'Thresholds': thresholds_rf
})

# ROC curve 
roc_plot = (ggplot(roc_data_rf, aes(x='False Positive Rate', y='True Positive Rate'))
            + geom_line()
            + labs(title=f'ROC Curve (AUC = {auc_rf:.2f})')
            + theme_bw())

roc_plot.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np

# Hyperparameter grid
rf_param_dist = {
    'n_estimators': np.arange(50, 301, 25).tolist(),
    'max_depth': [None] + list(np.arange(5, 31, 5)),
    'min_samples_split': np.arange(2, 21, 2).tolist(),
    'min_samples_leaf': np.arange(1, 11, 1).tolist()
}

# RandomizedSearchCV
rf_random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), 
                                    rf_param_dist, cv=5, 
                                    n_iter=10, random_state=42, 
                                    n_jobs=-1)
rf_random_search.fit(X_train_selected, y_train_resampled)

# best parameters
print("Best parameters for Random Forest:", rf_random_search.best_params_)

# cross-validation
cv_scores_rf = cross_val_score(rf_random_search.best_estimator_, X_train_selected, y_train, cv=10)
print("Cross-validation score for Random Forest:", cv_scores_rf.mean())


# In[ ]:


# best Random Forest model
best_rf_model = rf_random_search.best_estimator_

# best model to the entire training data
best_rf_model.fit(X_train_selected, y_train_resampled)

# predictions and evaluate on test data
y_pred_rf = best_rf_model.predict(X_test_selected)

# classification report and confusion matrix
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# In[ ]:


from imblearn.metrics import classification_report_imbalanced

# classification report
print(classification_report_imbalanced(y_test, y_pred_rf))


# In[ ]:


# Import necessary libraries
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred_rf)
print("\nBalanced Accuracy:", round(balanced_acc, 4))

# Calculate metrics for the "severe" class
target_names = {0: 'Not Severe', 1: 'Severe'}
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_rf, labels=[1], average=None)

print("\nMetrics for Severe Class (1):")
print(f"Precision: {precision[0]:.4f}")
print(f"Recall: {recall[0]:.4f}")
print(f"F1 Score: {f1_score[0]:.4f}")


# In[ ]:


import matplotlib.pyplot as plt

# fit Random Forest with the best parameters
best_rf_model = rf_random_search.best_estimator_

# fit the model on the training data
best_rf_model.fit(X_train_selected, y_train)

# feature importance from the Random Forest model
feature_importances_rf = best_rf_model.feature_importances_

# Pair features with their importance
feature_importance = pd.DataFrame({
    'Feature': X_train_selected.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

# sort the features by importance
sorted_idx_rf = feature_importances_rf.argsort()

# features
top_20_idx_rf = sorted_idx_rf[-7:]

# feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(7), feature_importances_rf[top_20_idx_rf], align='center')
plt.yticks(range(7), X_train_selected.columns[top_20_idx_rf])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# predicted probabilities for each model
y_pred_prob_rf = best_rf_model.predict_proba(X_test_selected)[:, 1]

# compute ROC curve and AUC for each model
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")

# customize plot
plt.title("ROC Random Forest - Right Eye", fontsize=20, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold")
plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

# customize legend
plt.legend(loc="lower right", fontsize=14, title="Legend", title_fontsize=16, frameon=True)
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# confusion matrices for all models and both eyes
confusion_matrices = {
    "Left Eye": {
        "Random Forest (Left Eye)": np.array([[37, 9], [9, 14]]),
        "SVM (Left Eye)": np.array([[36, 10], [9, 14]]),
        "XGBoost (Left Eye)": np.array([[38, 8], [11, 12]]),
        "KNN (Left Eye)": np.array([[39, 7], [13, 10]]),
    },
    "Right Eye": {
        "Random Forest (Right Eye)": np.array([[45, 1], [18, 5]]),
        "SVM (Right Eye)": np.array([[46, 0], [21, 2]]),
        "XGBoost (Right Eye)": np.array([[42, 4], [18, 5]]),
        "KNN (Right Eye)": np.array([[44, 2], [16, 7]])
    }
}

# grid
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# plot heatmaps
for row_idx, (eye, models) in enumerate(confusion_matrices.items()):
    for col_idx, (model, matrix) in enumerate(models.items()):
        ax = axes[row_idx, col_idx]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    annot_kws={"size": 14, "weight": "bold", "color": "black"},
                    ax=ax)
        ax.set_title(f"{model} ({eye})", fontsize=16, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=14, fontweight="bold")
        ax.set_xticklabels(['No', 'Yes'], fontsize=12, fontweight="bold")
        ax.set_yticklabels(['No', 'Yes'], fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.show()

