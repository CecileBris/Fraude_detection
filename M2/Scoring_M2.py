
# %% IMPORTS & SCORES

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statistics import mean

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import statsmodels.api as sm

# conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
# pip install xgboost
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# import statsmodels.api as sm
# pip install scorecardpy
import scorecardpy as sc

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),
           'Recall': make_scorer(recall_score), 'f1_score': make_scorer(f1_score),
           'Precision': make_scorer(precision_score)}

# %% PREPROCESS

path = r'C:\Users\ceecy\OneDrive - Université Paris-Dauphine\Economie\M2\S2_Scoring\autorisations.csv'
df = pd.read_csv(path, encoding='utf8')

# On met la date et l'heure dans une même variable qu'on convertit au format datetime
df[['Date', 'Heure']] = df[['Date', 'Heure']].astype(str)
df['Date_'] = (df['Date'] + ' ' + df['Heure'])

df.Date_ = df.Date_.apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S'))
df.drop(columns=['Date', 'Heure', 'dateheure'], inplace=True)

# Variables catégorielles :  'Pays', 'CodeRep', 'MCC', 'fraude'

df['fraude'].value_counts()

# Base pas dutout équilibré => SMOTE

# CHECK var intéressantes
# fig = plt.figure()
# for var in df.drop(columns=['fraude', 'CodeRep', 'Date_']).columns:
#     fig.add_subplot()
#     g = sns.FacetGrid(data=df, hue='fraude')
#     g.map(sns.histplot, var).add_legend()

# plt.show()

df["CodeRep"].loc[df["CodeRep"] != 0] = 1

# PREP data
df.drop(columns=['Date_'], inplace=True)

# %%% OUTLIERS DETECTION
# COOK'S DISTANCE : https://www.statology.org/cooks-distance-python/

X = df.drop(columns=['fraude'])
y = df['fraude']

# add constant to predictor variables
X = sm.add_constant(X)

# fit linear regression model
model = sm.OLS(y, X).fit()

# suppress scientific notation
np.set_printoptions(suppress=True)

# create instance of influence
influence = model.get_influence()

# obtain Cook's distance for each observation
cooks = influence.cooks_distance

# display Cook's distances
print(cooks)

# visualize Cook's distance & store it in the df to select outliers to supress
n = list(df.columns)
for i in n:
    # plt.scatter(df[i], cooks[0])
    # plt.xlabel(i)
    # plt.ylabel('Cooks Distance')
    # plt.show()
    df['Cooks_Distance'] = cooks[0]

seuil = 3*mean(df.Cooks_Distance)
df = df[df['Cooks_Distance'] < seuil]
df = df.drop(columns=['Cooks_Distance'])


# %%% Normalisation

# on part du df sans outliers
X = df.drop(columns=['fraude'])
y = df['fraude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Heatmap pour les correlations
plt.figure(figsize=[18, 7])
sns.heatmap(df.corr(), annot=True)
plt.show()

# Normalize our feature set
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# %%% DEPENDANCY DETECTION - LASSO

# Heatmap pour les correlations
plt.figure(figsize=[18, 7])
sns.heatmap(df.corr(), annot=True)
plt.show()

# LASSO pour réduction de dimensions

# On teste le LASSO pour différentes valeurs de alpha
alphas = np.linspace(0.001, 100, 10)
lasso = Lasso(max_iter=10000)
coefs = []
Erreurs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    y_pred = lasso.predict(X_test)
    Errors = y_test - y_pred
    Errors = Errors**2
    Erreurs.append(Errors.mean())

# Plot de l'activset
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title("Active set : Valeurs des coefficients en fonction de Lambda");

errors = pd.DataFrame(Erreurs)
best_alpha = alphas[errors[[0]].idxmin()]

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train, y_train)
Lasso(alpha=best_alpha)
""" 0.01
"""

print(list(zip(lasso_best.coef_, X_train)))
"""
[(4.809798210295322e-10, 'Carte'), (8.54485351602769e-07, 'Pays'), (-0.0, 'CodeRep'),
 (-1.547185365046084e-11, 'MCC'), (-0.0, 'Montant'), (-0.0, 'FM_Velocity_Condition_3'),
 (-0.0, 'FM_Velocity_Condition_6'), (-0.0, 'FM_Velocity_Condition_12'), (-0.0, 'FM_Velocity_Condition_24'),
 (0.0, 'FM_Sum_3'), (0.0, 'FM_Sum_6'), (-3.625393920494065e-08, 'FM_Sum_12'),
 (-1.3174836371082496e-07, 'FM_Sum_24'),  (-0.0, 'FM_Redondance_MCC_3'), (-0.0, 'FM_Redondance_MCC_6'),
 (-0.0, 'FM_Redondance_MCC_12'), (-0.0, 'FM_Redondance_MCC_24'), (-0.0, 'FM_Difference_Pays_3'),
 (-0.0, 'FM_Difference_Pays_6'), (-0.0, 'FM_Difference_Pays_12'), (-0.0, 'FM_Difference_Pays_24')]
"""

""" 0.001
[(5.059414747564177e-10, 'Carte'), (9.601345686516815e-07, 'Pays'), (-0.0, 'CodeRep'),
 (-4.025551201352077e-09, 'MCC'), (-1.1268813038844393e-07, 'Montant'), (-0.0, 'FM_Velocity_Condition_3'),
 (-0.0, 'FM_Velocity_Condition_6'), (-0.0, 'FM_Velocity_Condition_12'), (-0.0, 'FM_Velocity_Condition_24'),
 (4.915777708116309e-07, 'FM_Sum_3'), (-0.0, 'FM_Sum_6'), (-4.791028602110329e-07, 'FM_Sum_12'),
 (-1.197747905673279e-07, 'FM_Sum_24'), (-0.0, 'FM_Redondance_MCC_3'), (-0.0, 'FM_Redondance_MCC_6'),
 (-0.0, 'FM_Redondance_MCC_12'), (-0.0, 'FM_Redondance_MCC_24'), (-0.0, 'FM_Difference_Pays_3'),
 (-0.0, 'FM_Difference_Pays_6'), (-0.0, 'FM_Difference_Pays_12'), (-0.0, 'FM_Difference_Pays_24')]
"""

# Make a plot of coefficients
plt.plot(range(len(list(X_train))), list(lasso_best.coef_))
plt.xticks(range(len(list(X_train))), list(X_train), rotation=60)
plt.title('Représentations des coefficients')
plt.xlabel(None)
plt.ylabel('Coefficients')
plt.figure(dpi=175)
plt.show()

y_pred = lasso_best.predict(X_test)
Errors = y_test - y_pred
Errors_carré = Errors**2
Errors_carré.mean()

# Lasso regression object
lasso = Lasso(alpha=best_alpha, normalize=True)

# Fit to data
lasso.fit(X_train, y_train)

# Printed results
lasso_coef = lasso.coef_
print(lasso_coef)
print(list(X_train))

# Make a plot of coefficients
plt.plot(range(len(list(X_train))), lasso_best.coef_)
plt.xticks(range(len(list(X_train))), list(X_train), rotation=60)
plt.title('Représentations des coefficients')
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.figure(dpi=175)
plt.show()

# %%%% Si on applique la sélection du LASSO : lancer cette cellule /!\

df = df[["Carte", "Pays", "MCC", "Montant", "FM_Sum_3", "FM_Sum_12", "FM_Sum_24", "fraude"]]

# %%% DEPENDANCY DETECTION - PCA

pca = PCA(random_state=42)
X_train_pca = pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


""" Explained variance -----------------------------------------------------"""

# Returns the variance caused by each of the principal components

explained_variance = pca.explained_variance_ratio_
explained_variance*100
print(explained_variance*100)


# Interpretation : first principal component is responsible for XX% variance.

# GRAPH Variance expliquée

fig = plt.figure(figsize=(12, 8))
plt.plot(explained_variance*100)
plt.title('Fraude – Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.show()

# Making the screeplot
# plotting the cumulative variance against the number of components

# np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

fig = plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.vlines(x=10, ymax=1, ymin=0, colors="r", linestyles="--")
# plt.hlines(y=.6984, xmax=17, xmin=0, colors="g", linestyles="--")
plt.title('Fraude – Cumulative Variance')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


""" Components -------------------------------------------------------------"""

components = pd.DataFrame({'PC1': pca.components_[0], 'PC2': pca.components_[1], 'Feature': X.columns})
components

# Graph of PC
fig = plt.figure(figsize=(12, 8))
plt.scatter(components.PC1, components.PC2)
plt.title('Category – Principal Component')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(components.Feature):
    plt.annotate(txt, (components.PC1[i], components.PC2[i]))
plt.show()


# %% REEQUILIBRAGE

# Recalibrage des 0, 1 => Smoth


def creation_des_classes():
    """
    CREATION DES CLASSES PAR "CHIMERGE"
    """
    classes = sc.woebin(df, y="fraude", positive=1, method="chimerge")
    return classes


classes = creation_des_classes()
sc.woebin_plot(classes)


def split(train_size):
    train_set, test_set = sc.split_df(df, y="fraude", ratio=train_size, seed=186).values()
    print('Le train_size est :', train_size)
    print("\n")
    print('Shape train_set', train_set.shape)
    print('Shape test_set', test_set.shape)
    return train_set, test_set


train_set, test_set = split(0.75)


# CONVERSION DES CLASSES DES VARIABLES DES TRAIN ET test SETS EN WOE
def conversion_woe():
    print("Cette opération peut prendre quelques minutes...\n")
    train_woe = sc.woebin_ply(train_set, classes)
    test_woe = sc.woebin_ply(test_set, classes)
    # CREATION DES X ET y DES TRAIN ET test SETS
    y_train = train_woe.loc[:, "fraude"]
    x_train = train_woe.loc[:, train_woe.columns != "fraude"]
    y_test = test_woe.loc[:, "fraude"]
    x_test = test_woe.loc[:, train_woe.columns != "fraude"]
    return y_train, x_train, y_test, x_test, train_woe, test_woe


y_train, x_train, y_test, x_test, train_woe, test_woe = conversion_woe()

# pipeline_knn = Pipeline([
#     ('imputer', SimpleImputer(copy=False)),
#     ('scaler', StandardScaler(copy=False)),
#     ('lda', LinearDiscriminantAnalysis()),
#     ('model', KNeighborsClassifier(n_jobs=-1))
# ])

# # Mean imputation by default
# param_grid_knn = {
#     'model__n_neighbors': np.arange(5, 50, 5)
# }

# grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring=scoring, refit='AUC',
#                         n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

# grid_knn.fit(x_train, y_train)

# grid_knn.best_score_
# grid_knn.best_params_

"45"

def over_sampling():
    # ON VA FAIRE DE MANIÈRE AUTO
    print("On va rééchantillonner le train set...\n")
    smote = SMOTE(random_state=0, k_neighbors=45)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    print(smote.get_params(deep=True))
    print("\n")
    print("La nouvelle proportion de 0 et 1 de la target resamplée est: %s" % Counter(y_train_smote))
    print("\n")
    print(y_train_smote.value_counts(normalize=True))
    return x_train_smote, y_train_smote


x_train_smote, y_train_smote = over_sampling()

x_train_smote.to_csv(r"C:\Users\ceecy\OneDrive - Université Paris-Dauphine\Economie\M2\S2_Scoring\x_train_smote.csv",
                     sep='|', index=False)

y_train_smote.to_csv(r"C:\Users\ceecy\OneDrive - Université Paris-Dauphine\Economie\M2\S2_Scoring\y_train_smote.csv",
                     sep='|', index=False)

y_test.to_csv(r"C:\Users\ceecy\OneDrive - Université Paris-Dauphine\Economie\M2\S2_Scoring\y_test.csv",
                     sep='|', index=False)
x_test.to_csv(r"C:\Users\ceecy\OneDrive - Université Paris-Dauphine\Economie\M2\S2_Scoring\x_test.csv",
                     sep='|', index=False)

# %% LOGISTIC REGRESSION

# Mean imputation by default
pipeline_sgdlogreg = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

param_grid_sgdlogreg = {
    'model__penalty': ['l1', 'l2']
}

grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg,
                              scoring='recall', n_jobs=1, pre_dispatch=1, cv=5, verbose=1,
                              return_train_score=False)

grid_sgdlogreg.fit(x_train_smote, y_train_smote)

grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
print(grid_sgdlogreg.best_params_)
print(grid_sgdlogreg.best_score_, '- Logistic regression')


def modèle():
    # LE MODÈLE FINAL QUI SERA APPLIQUÉ AU TEST SET
    print("Voici notre meilleur modèle \n")
    m = LogisticRegression(penalty='l2', C=5, solver='saga', max_iter=100,
                           random_state=1).fit(x_train_smote, y_train_smote)
    print("Le modèle est :", m)
    print("m.fit(X_train_pca, y_train_smote) \n")
    # PREDICTIONS À PARTIR DU SMOTE
    test_proba_m = m.predict_proba(x_test)[:, 1]
    train_proba_m = m.predict_proba(x_train_smote)[:, 1]
    y_pred_m = m.predict(x_test)
    test_proba_m_b = test_proba_m > 0.5
    # SCORES DU 2E MODÈLE RESAMPLÉ
    train_score_m = m.score(x_train_smote, y_train_smote)
    test_score_m = m.score(x_test, y_test)
    print("train score :", train_score_m)
    print("test score :", test_score_m)
    # CLASSIFICATION REPORT
    print("\n")
    class_report_m = classification_report(y_test, test_proba_m_b)
    print("classification_report :\n", class_report_m)
    return m, train_proba_m, test_proba_m, y_pred_m


m, train_proba_m, test_proba_m, y_pred_m = modèle()

# %% Xgboost

params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.5, 1, 5]}

grid_xgb = GridSearchCV(xgb.XGBClassifier(n_jobs=-1, objective='binary:logistic'), params, n_jobs=1, cv=5,
                        scoring='recall')
grid_xgb.fit(x_train_smote, y_train_smote)

grid_xgb.best_score_
grid_xgb.best_params_

print(grid_xgb.best_score_)
print(grid_xgb.best_params_)

xgb_pred = grid_xgb.predict(x_test)
xgb_proba = grid_xgb.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# %% KNN

pipeline_knn = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('scaler', StandardScaler(copy=False)),
    ('lda', LinearDiscriminantAnalysis()),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

# Tunning : The 'k' in k-nearest neighbors
param_grid_knn = {
    'model__n_neighbors': np.arange(5, 50, 5)
}

grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='recall',
                        n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_knn.fit(x_train_smote, y_train_smote)

print(grid_knn.best_score_)
print(grid_knn.best_params_)

y_pred_KNN = grid_knn.predict(x_test)
y_proba_KNN = grid_knn.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred_KNN))
print(grid_knn.best_score_, '- KNN')

# %% RANDOM FOREST

pipeline_rfc = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])

# The number of randomized trees to build : 50
param_grid_rfc = {
    'model__n_estimators': np.arange(20, 60, 10)
}

grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='recall',
                        n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_rfc.fit(x_train_smote, y_train_smote)

print(grid_rfc.best_score_)

y_pred_RF = grid_rfc.predict(x_test)
y_proba_RF = grid_rfc.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred_RF))
print(grid_rfc.best_score_, '- Random forest')


# %% Voting

clf1 = LogisticRegression(penalty='l1', C=5, solver='saga', max_iter=100, random_state=1)
clf2 = grid_xgb
clf3 = grid_knn
clf4 = grid_rfc

estimators = [('lr', clf1), ('xgb', clf2), ('knn', clf3), ('rf', clf4)]

vclf = VotingClassifier(estimators, voting='soft')

# %% EVAL MODEL
# def courbe de lift function


def plot_lift_curve(y_val, y_pred, step=0.01):
    """
    Parameters
    ----------
    aux_lift : Define an auxiliar dataframe to plot the curve
    Create a real and predicted column for our new DataFrame and assign values :
    y_val : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    step : TYPE, optional
        DESCRIPTION. The default is 0.01.

    Returns
    -------
    None.

    """
    aux_lift = pd.DataFrame()
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    # Order the values for the predicted probability column:
    aux_lift.sort_values('predicted', ascending=False, inplace=True)
    # Create the values that will go into the X axis of our plot
    x_val = np.arange(step, 1+step, step)
    # Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    # Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []
    # Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift)))
        # The ceil function returns the closest integer bigger than our number
        data_here = aux_lift.iloc[:num_data, :]
        # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)
    # Plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40, 40)
    axis.plot(x_val, y_v, 'g-', linewidth=3, markersize=5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title('Lift Curve')
    plt.show()


# %%% Score print

print(grid_sgdlogreg.best_score_, '- Logistic regression')
print(grid_knn.best_score_, '- KNN')
print(grid_rfc.best_score_, '- Random forest')
print(xgb_clf.score(x_test, y_test), '- xgBoost')

# %%% Logistic regression - PSI


def points_par_classe(base, pdo):
    # CREATION DE LA GRILLE SCORE À PARTIR DES CLASSES, DU MODÈLE M ET CALIBRAGE SUR "base" POINTS
    points_par_classe = sc.scorecard(classes, m, xcolumns=x_train.columns, points0=base,
                                     pdo=pdo)
    return points_par_classe


points_par_classe = points_par_classe(base=1000, pdo=50)


def scores():
    # CALCUL DES SCORES TOTAUX DANS LE TRAIN SET
    train_score = sc.scorecard_ply(train_set, points_par_classe, print_step=0)
    # CALCUL DES SCORES DANS LE TEST SET
    test_score = sc.scorecard_ply(test_set, points_par_classe, print_step=0)
    return train_score, test_score


train_score, test_score = scores()
train_score

# POPULATION STABILITY INDEX (PSI)
sc.perf_psi(score={'train': train_score, 'test': test_score},
            label={'train': y_train, 'test': y_test},
            return_distr_dat=True)


# %%% ROC-AUC

# train_perf_ROC = sc.perf_eva(y_train_smote, train_proba_m, title="train", positive=1, plot_type=["roc", "pr"])
test_perf_ROC = sc.perf_eva(y_test, test_proba_m, title="test", plot_type=["roc", "pr"])

test_perf_ROC_RF = sc.perf_eva(y_test, y_proba_RF, title="test", plot_type=["roc", "pr"])

test_perf_ROC_KNN = sc.perf_eva(y_test, y_proba_KNN, title="test", plot_type=["roc", "pr"])

test_perf_ROC_XGB = sc.perf_eva(y_test, xgb_proba, title="test", plot_type=["roc", "pr"])

# %%% KOLMOGOROV - SMIRNOV & Lift
# train_perf_KS = sc.perf_eva(y_train_smote, train_proba_m, title="train", positive=1, plot_type=["ks"])
test_perf_KS = sc.perf_eva(y_test, test_proba_m, title="test", plot_type=["ks", 'lift'])

test_perf_ROC_RF = sc.perf_eva(y_test, y_proba_RF, title="test", plot_type=["ks", 'lift'])

test_perf_ROC_KNN = sc.perf_eva(y_test, y_proba_KNN, title="test", plot_type=["ks", 'lift'])

test_perf_ROC_XGB = sc.perf_eva(y_test, xgb_proba, title="test", plot_type=["ks", 'lift'])

# Le random forest est au top ! On choisira ce modèle

plot_lift_curve(y_test, test_proba_m)
plot_lift_curve(y_test, y_proba_RF)
plot_lift_curve(y_test, y_proba_KNN)
plot_lift_curve(y_test, xgb_proba)

# %%% results

"""
runcell('LOGISTIC REGRESSION', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
Fitting 5 folds for each of 6 candidates, totalling 30 fits
0.7932326761639155 - Logistic regression
Voici notre meilleur modèle 

C:\Users\ceecy\anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:352: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
La modèle est : LogisticRegression(C=5, random_state=0, solver='saga')
m.fit(X_train_smote, y_train_smote) 

train score : 0.7498242754426975
test score : 0.7457297403206281


classification_report :
               precision    recall  f1-score   support

           0       1.00      0.75      0.85    285484
           1       0.00      0.70      0.01       330

    accuracy                           0.75    285814
   macro avg       0.50      0.72      0.43    285814
weighted avg       1.00      0.75      0.85    285814

runcell('Xgboost', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
[[     0 285484]
 [     0    330]]
C:\Users\ceecy\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

           0       0.00      0.00      0.00    285484
           1       0.00      1.00      0.00       330

    accuracy                           0.00    285814
   macro avg       0.00      0.50      0.00    285814
weighted avg       0.00      0.00      0.00    285814

runcell('KNN', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
Fitting 5 folds for each of 9 candidates, totalling 45 fits
[[125139 160345]
 [    27    303]]
0.8558299237415726 - KNN

runcell('RANDOM FOREST', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
Fitting 5 folds for each of 8 candidates, totalling 40 fits
[[ 36852 248632]
 [     0    330]]
0.9486214705171351 - Random forest

"""

"""
LASSO
runcell('LOGISTIC REGRESSION', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
Fitting 5 folds for each of 6 candidates, totalling 30 fits
0.7488378516111702 - Logistic regression
Voici notre meilleur modèle 

La modèle est : LogisticRegression(C=5, random_state=0, solver='saga')
m.fit(X_train_smote, y_train_smote) 

train score : 0.7134201019552714
test score : 0.6805194986949554


classification_report :
               precision    recall  f1-score   support

           0       1.00      0.68      0.81    285484
           1       0.00      0.67      0.00       330

    accuracy                           0.68    285814
   macro avg       0.50      0.67      0.41    285814
weighted avg       1.00      0.68      0.81    285814


runcell('Xgboost', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
[[     0 285484]
 [     0    330]]
C:\Users\ceecy\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

           0       0.00      0.00      0.00    285484
           1       0.00      1.00      0.00       330

    accuracy                           0.00    285814
   macro avg       0.00      0.50      0.00    285814
weighted avg       0.00      0.00      0.00    285814

C:\Users\ceecy\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\ceecy\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
  
  runcell('KNN', 'C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S2_Scoring/Scoring_TSO_BRISSARD.py')
  Fitting 5 folds for each of 9 candidates, totalling 45 fits
  C:\Users\ceecy\anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py:702: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
    warnings.warn(
  [[ 86002 199482]
   [    24    306]]
  0.8216477525005279 - KNN
  
  """
