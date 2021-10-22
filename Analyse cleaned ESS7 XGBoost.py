# analyse ESS7 data cleaned 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import xgboost as xgb
from sklearn import metrics
import shap
from sklearn.inspection import plot_partial_dependence
import pickle 


data = pd.read_csv('https://raw.githubusercontent.com/Thomas-Richardson/Blog_post_data/main/ESS7_cleaned.csv')
#data.shape
#data.columns
data = data.drop(columns = 'Unnamed: 0')
#data.head(20)

Richie_rand_seed = 14

X = data.drop(columns = 'happy')
y = data.loc[:,'happy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = Richie_rand_seed)

XGB_model = xgb.XGBRegressor(objective = 'reg:squarederror', alpha = 10)
#XGB_model.fit(X_train, y_train)

# randomized search CV

random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 5)],
    'learning_rate':np.linspace(start = 0.01, stop = 0.2, num = 10),
    'max_depth' : [int(x) for x in np.linspace(3, 10, num = 4)],
    'subsample': np.linspace(0.5,1,5),
    'colsample_by_tree': np.linspace(0.5,1,5)
               }

xgb_random = RandomizedSearchCV(estimator = XGB_model, param_distributions = random_grid, n_iter = 100, cv = 5, n_jobs = -1, verbose=2, random_state=Richie_rand_seed).fit(X_train,y_train)
xgb_random.best_params_
XGB_model= xgb_random
pickle.dump(xgb_random, open("XGB_random_iter10_cv3.dat",'wb'))
#XGB_model = pickle.load(open("XGB_random_iter10_cv3.dat", "rb"))

y_pred = XGB_model.predict(X_test)

#rng = np.random.default_rng(Richie_rand_seed)
#rng.choice(y_pred,5)

y_pred = y_pred.round()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(2))
print('R squared:', metrics.r2_score(y_test, y_pred).round(2))

round(sum(y_pred == y_test) / len(y_test),2)
round(sum( abs(y_pred - y_test) <= 1) / len(y_test),2)
round(sum( abs(y_pred - y_test) >= 4) / len(y_test),2)

y_pred_null = np.repeat(y_train.median(), len(y_pred))
print('MAE of null model:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred_null)).round(2))
print('RMSE of null model:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_null)).round(2))

round(((1.06-1.15)/1.15 )*100,1)
round(((1.49-1.93)/1.93 )*100,1)

feature_importances = XGB_model.feature_importances_

xgb_results = pd.DataFrame({'Feature':list(X.columns), 'Feature importance':list(feature_importances)})
xgb_results_sorted = xgb_results.sort_values('Feature importance', ascending = False)

# Partial dependence plots
print(xgb_results.Feature[0:10])
print(np.where(np.isin(X.columns, xgb_results.Feature[0:10])))
plot_partial_dependence(estimator = XGB_model,X = X_train, features = [],feature_names=X_train.columns.tolist())

# Feature importance plots
plt.rcParams['figure.figsize'] = [20, 8]
sns.barplot(y = 'Feature', x = 'Feature importance', data = xgb_results_sorted.iloc[0:15,:])
plt.show()

#=====================================================================================================
# Shap values
my_model_explainer = shap.TreeExplainer(XGB_model) # Create object that can calculate shap values
shap_values = my_model_explainer.shap_values(X_test) # Calculate Shap values

X_sample = X_test.sample(100)
shap_values = my_model_explainer.shap_values(X_sample) 

shap.summary_plot(shap_values, X_sample)

obs = 0
shap_plot = shap.force_plot(my_model_explainer.expected_value, 
    shap_values[obs], features = X_test.iloc[obs], 
    feature_names = X_test.columns,
    matplotlib=True, plot_cmap=['#77dd77', '#f99191'])

shap.dependence_plot('age',shap_values, X_sample)
shap.dependence_plot('age',shap_values, X_sample, interaction_index=None)