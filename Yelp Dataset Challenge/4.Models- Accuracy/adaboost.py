#-----------------------------------------AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaBoostClassifier = AdaBoostClassifier()
tuned_parameters = [{'n_estimators':[220,230,225],'learning_rate':[0.5],'algorithm':['SAMME','SAMME.R'],'random_state':[10]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(adaBoostClassifier, tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))
    
    print()


