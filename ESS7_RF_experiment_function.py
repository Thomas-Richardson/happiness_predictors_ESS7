# analyse ESS7 data cleaned 

def RF_pipeline(data):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.preprocessing import OrdinalEncoder

    Richie_rand_seed = 14
    
    X = data.drop(columns = ['happy'])
    y = data.loc[:,'happy']

    X.country = OrdinalEncoder().fit_transform(X[['country']])
    #X = pd.get_dummies(X, columns = ['country'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = Richie_rand_seed)

    RF_model = RandomForestRegressor(n_estimators = 500, random_state = Richie_rand_seed,n_jobs = -1)
    RF_model.fit(X_train, y_train)

    y_pred = RF_model.predict(X_test)

    rng = np.random.default_rng(Richie_rand_seed)
    rng.choice(y_pred,5)

    y_pred = y_pred.round()

    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred).round(2))  
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(2)
    R_sq =  metrics.r2_score(y_test, y_pred).round(2)

    #print("model exactly right:", round(sum(y_pred == y_test) / len(y_test),2))
    #print("model within 1", round(sum( abs(y_pred - y_test) <= 1) / len(y_test),2))
    #print("model dead wrong", round(sum( abs(y_pred - y_test) >= 4) / len(y_test),2))

    y_pred_null = np.repeat(y_train.median(), len(y_pred))
    #print('MAE of null model:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred_null)).round(2))
    RMSE_null =  np.sqrt(metrics.mean_squared_error(y_test, y_pred_null)).round(2)

    return RMSE, R_sq

"""
Ordinal countries
Root Mean Squared Error: 1.47
R squared: 0.29
RMSE of null model: 1.81

Dummy countries
Root Mean Squared Error: 1.47
R squared: 0.29
RMSE of null model: 1.81

Going forward with ordinal countries as faster

5 imputed income decile
Root Mean Squared Error: 1.49
RMSE of null model: 1.83
R squared: 0.28

-1 imputed income decile
Root Mean Squared Error: 1.49
RMSE of null model: 1.83
R squared: 0.28

Going forward without income decile nulls
"""



