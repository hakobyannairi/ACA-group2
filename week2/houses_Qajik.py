import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
import sys
import warnings
warnings.filterwarnings("ignore")
if len(sys.argv) != 3:
    print("Usage: python3 houses.py train_path test_path")
    exit(-1)

### Preprocessing
df1 = pd.read_csv(sys.argv[1])
y = df1['price']
train_len = df1.shape[0]
df1 = df1.drop(['price'], axis=1)
df2 = pd.read_csv(sys.argv[2])
df = pd.concat((df1, df2), axis=0)

df = df.drop(['url'], axis=1)
df = df.loc[:,~df.columns.str.contains('^Unnamed')]
if len(list(set(df['region'].tolist()))) == 1:
    df = df.drop(['region'], axis=1)

isfirst = []
islast = []
c = []
m = []
for index, row in df[['floor', 'max_floor']].iterrows():
    floor = row[0]
    max_floor = row[1]
    g = 0
    e = 0
    mi_harkani = 0
    if floor == 1:
        g = 1
    if floor == max_floor:
        e = 1
    if floor == 1 and max_floor == 1:
        mi_harkani = 1
    c += [2 - (floor/max_floor - 0.3)**2]
    isfirst += [g]
    islast += [e]
    m += [mi_harkani]
        

df['floor_dr'] = c
df['isfirst'] = isfirst
df['islast'] = islast
df['mi_harkani'] = m
df['area*num_rooms'] = df['area'] / (df['num_rooms'] + df['num_bathrooms'])

data = df.select_dtypes(['number'])

X = pd.get_dummies(df.select_dtypes(exclude=['number']))
X = np.hstack((data, X))

poly = PolynomialFeatures(2)
print(X)
X = poly.fit_transform(X)
X_train = X[0:train_len]
X_test = X[train_len:]

### Feature selection
lsvc = LinearSVC(C=300, penalty="l1", dual=False).fit(X_train, y)
model = SelectFromModel(lsvc, prefit=True, threshold=1e-27, max_features=5000)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

### Model
estimators = [
    ('lr1', RidgeCV()),
    ('lr2', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('lr3', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)),
    ('lr4', DecisionTreeRegressor(random_state=0))
    ]
xgb_r = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(311)
)

print('started fitting')
xgb_r.fit(X_train, y)
print('model fitted')
y_pred = xgb_r.predict(X_test)

### Output
file = open('y_pred.csv', 'w')
file.write('price\n')
for c in y_pred:
    file.write(str(c) + '\n')
file.close()
