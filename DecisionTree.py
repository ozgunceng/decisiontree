##Generate a moons dataset using make_moons(n_samples=10000, noise=0.4). 
from sklearn.datasets import make_moons 
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42) 
##Split it into a training set and a test set using train_test_split(). 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
#Use grid search with cross-validation (with the help of the GridSearchCV class) to find good hyperparameter values for a 
DecisionTreeClassifier. Hint: try various values for max_leaf_nodes. 
from sklearn.model_selection import GridSearchCV 
from sklearn.tree import DecisionTreeClassifier 
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]} 
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1, cv=3) 
grid_search_cv.fit(X_train, y_train) 
grid_search_cv.best_estimator_ 
from sklearn.metrics import accuracy_score 
y_pred = grid_search_cv.predict(X_test) 
accuracy_score(y_test, y_pred) 