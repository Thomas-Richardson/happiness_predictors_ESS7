# analyse ESS7 data cleaned 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance
import shap

data = pd.read_csv('https://raw.githubusercontent.com/Thomas-Richardson/Blog_post_data/main/ESS7_cleaned.csv')
data.shape
data.columns
data = data.drop(columns = 'Unnamed: 0')
data.head(20)

Richie_rand_seed = 14

X = data.drop(columns = 'happy')
y = data.loc[:,'happy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = Richie_rand_seed)

RF_model = RandomForestRegressor(n_estimators = 1000, random_state = Richie_rand_seed)
RF_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)

rng = np.random.default_rng(Richie_rand_seed)
rng.choice(y_pred,5)

y_pred = y_pred.round()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(2))
print('R squared:', metrics.r2_score(y_test, y_pred).round(2))

sum(y_pred == y_test) / len(y_test)
sum( abs(y_pred - y_test) <= 1) / len(y_test)
sum( abs(y_pred - y_test) >= 4) / len(y_test)

y_pred_null = np.repeat(y_train.median(), len(y_pred))
print('MAE of null model:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred_null)).round(2))
print('RMSE of null model:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_null)).round(2))

round(((1.07-1.15)/1.15 )*100,1)
round(((1.49-1.93)/1.93 )*100,1)

'''feature_importances = RF_model.feature_importances_
#permuted_feature_importances = permutation_importance(RF_model, X_test, y_test)
rf_results = pd.DataFrame({'Feature':list(X.columns), 'Feature importance':list(feature_importances)})
sorted_idx = permuted_importances.importances_mean.argsort()
#rf_results2 = pd.DataFrame({'Feature':X_test.columns[sorted_idx], 'Permuted feature importance':abs(permuted_importances.importances_mean[sorted_idx])})
#rf_results_merged = pd.merge(rf_results,rf_results2, on = 'Feature', how ='inner')

#plt.rcParams['figure.figsize'] = [30, 40] # set figure size parameters
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (30,40))
ax1.set_title('Feature importance the standard way')
sns.barplot(ax = ax1, y = 'Feature', x = 'Standard feature importance', data = rf_results_merged.sort_values('Standard feature importance', ascending = False))
ax2.set_title('Feature importance through the permutation approach')
sns.barplot(ax=  ax2, y = 'Feature', x = 'Permuted feature importance', data = rf_results_merged.sort_values('Permuted feature importance', ascending = False))
f.tight_layout()'''

plt.rcParams['figure.figsize'] = [20, 8]
feature_importances = RF_model.feature_importances_

rf_results = pd.DataFrame({'Feature':list(X.columns), 'Feature importance':list(feature_importances)})
rf_results_sorted = rf_results.sort_values('Feature importance', ascending = False)
sns.barplot(y = 'Feature', x = 'Feature importance', data = rf_results_sorted.iloc[0:15,:])
plt.show()

my_model_explainer = shap.TreeExplainer(RF_model) # Create object that can calculate shap values
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