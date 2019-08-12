import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

def target_encoder_regularized(train, cols_encode, target, folds=5):
    """
    Mean regularized target encoding based on kfold
    """
	
    kf = KFold(n_splits=folds, random_state=1)

    for col in cols_encode:
        global_mean = train[target].mean()

        for train_index, test_index in kf.split(train):
            mean_target = train.iloc[train_index].groupby(col)[target].mean()
            train.loc[test_index, col + "_mean_enc"] = train.loc[test_index, col].map(mean_target)
        train[col + "_mean_enc"] = train[col + "_mean_enc"].fillna(global_mean)
    return train
	
def log_features(train, cols):

	for col in cols:
		#reemplazo los 0 por el minimo
		train.loc[train[col] == 0, col] = train.loc[train[col] != 0, col ].min()
		#calculo logaritmo
		train[col + '_log'] = np.log(train[col])
	return train
	

def polynomialFeatures(n, df, lista, interaction_only = False):
	#features para interacciones
	X_poly = df[lista]
	#fit_transform del polinomial features
	poly = PolynomialFeatures(n,interaction_only = interaction_only)
	X_poly = pd.DataFrame(poly.fit_transform(X_poly),columns=poly.get_feature_names(X_poly.columns))
	#borro las features que ya están en el df
	X_poly = X_poly.drop(['1'] + lista ,axis = 1)
	#reseteo indices para poder concatenar
	df = df.reset_index(drop='True')
	df = pd.concat([df,X_poly],axis = 1)
	return df
	