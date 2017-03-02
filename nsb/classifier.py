from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import model_selection


def calculate_scores(model, features, target):
    return model_selection.cross_val_score(model, features, target, cv=10, scoring='roc_auc')


def calibrate(model, features, target, param_grid):
    gs = model_selection.GridSearchCV(model, param_grid=param_grid, cv=10, n_jobs=-1)
    gs.fit(features, target)
    return gs.best_estimator_


# -----------------------------------------------------------------------------
# decision tree
# -----------------------------------------------------------------------------


def build_decision_tree():
    return tree.DecisionTreeClassifier()


def get_param_grid_of_decision_tree():
    return {'max_depth': [4, 5, 6, None], 'max_features': [2, None], 'min_samples_split': [2, 8, 16, 32],
            'min_samples_leaf': [2, 8, 16, 32], 'max_leaf_nodes': [10, 50, 100]}


def calibrate_decision_tree(features, target, model=build_decision_tree(),
                            param_grid=get_param_grid_of_decision_tree()):
    return calibrate(model, features, target, param_grid)


# -----------------------------------------------------------------------------
# svc
# -----------------------------------------------------------------------------

def build_svc():
    return svm.SVC()


def get_param_grid_of_svc():
    return {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],  # , 'poly', 'sigmoid', 'linear'],
            'gamma': [0.5, 2, 16, 64]}


def calibrate_svc(features, target, model=build_svc(), param_grid=get_param_grid_of_svc()):
    return calibrate(model, features, target, param_grid)


# -----------------------------------------------------------------------------
# random forest
# -----------------------------------------------------------------------------

def build_random_forest():
    return ensemble.RandomForestClassifier()


def get_param_grid_of_random_forest():
    return {'n_estimators': [50, 100], 'max_depth': [4, 5, 6, None], 'max_features': [2, None],
            'max_leaf_nodes': [50, None]}


def calibrate_random_forest(features, target, model=build_random_forest(),
                            param_grid=get_param_grid_of_random_forest()):
    return calibrate(model, features, target, param_grid)


# -----------------------------------------------------------------------------
# extra trees
# -----------------------------------------------------------------------------

def build_extra_trees():
    return ensemble.ExtraTreesClassifier()


def get_param_grid_of_extra_trees():
    return get_param_grid_of_random_forest()


def calibrate_extra_trees(features, target, model=build_extra_trees(), param_grid=get_param_grid_of_extra_trees()):
    return calibrate(model, features, target, param_grid)


# -----------------------------------------------------------------------------
# ada boost
# -----------------------------------------------------------------------------

def build_ada_boost():
    return ensemble.AdaBoostClassifier(build_decision_tree())


def get_param_grid_of_ada_boost():
    return {'base_estimator__max_depth': [4, 5, 6, None], 'base_estimator__max_features': [2, None],
            'base_estimator__min_samples_split': [2, 8, 32], 'base_estimator__min_samples_leaf': [2, 8, 32],
            'base_estimator__max_leaf_nodes': [50, None], 'learning_rate': [0.5, 4], 'n_estimators': [100]}


def calibrate_ada_boost(features, target, model=build_ada_boost(), param_grid=get_param_grid_of_ada_boost()):
    return calibrate(model, features, target, param_grid)


# -----------------------------------------------------------------------------
# gradient boost
# -----------------------------------------------------------------------------

def build_gradient_boost():
    return ensemble.GradientBoostingClassifier()


def get_param_grid_of_gradient_boost():
    return {'n_estimators': [50, 100], 'max_depth': [4, 5, 6, None], 'max_features': [2, None],
            'max_leaf_nodes': [50, None], 'learning_rate': [0.1, 0.5, 1.0, 2.0]}


def calibrate_gradient_boost(features, target, model=build_gradient_boost(),
                             param_grid=get_param_grid_of_gradient_boost()):
    return calibrate(model, features, target, param_grid)
