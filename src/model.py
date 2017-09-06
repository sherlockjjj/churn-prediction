import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess
from utils import plot_bar
from sklearn.metrics import confusion_matrix, recall_score, make_scorer

if __name__ == '__main__':
    train = pd.read_csv('../data/churn_train.csv')
    test = pd.read_csv('../data/churn_test.csv')

    #drop all the missing values
    train = preprocess(train)
    cols = [u'avg_rating_by_driver', u'avg_rating_of_driver',
       u'avg_surge', u'surge_pct', u'trips_in_first_30_days',
       u'luxury_car_user', u'weekday_pct',
       u'city_Astapor', u'city_King\'s Landing',
       u'phone_Android', u'phone_iPhone', 'churn']

    y = train.pop('churn').values
    X = train.values
    # test = preprocess(test)
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=8, max_features='log2')
    # rf.fit(X_train, y_train)
    # scores = cross_val_score(rf, X_train, y_train, cv=3)
    #
    # test_score = rf.score(X_test, y_test)
    # confusion_matrix = confusion_matrix(rf.predict(X_test), y_test)
    rf = RandomForestClassifier()
    recall = make_scorer(recall_score)
    param_grid = dict(n_estimators=[20, 40, 100, 200],
                  max_depth = [3,4,6,8],
                  max_features = ['sqrt', 'log2', None],
                  bootstrap = [True, False])

    grid_search = GridSearchCV(rf, param_grid=param_grid, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)
    print("Grid scores on development set:")
    # print means
    rf.fit(X_train, y_train)
    print recall_score(rf.predict(X_test), y_test)
    # print test_score

    # combo = zip(rf.feature_importances_, train.columns)
    # combo = np.array(sorted(combo, reverse=True))
    # plot_bar(combo[:,1], combo[:,0])
