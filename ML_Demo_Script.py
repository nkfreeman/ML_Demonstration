# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:44:55 2019

@author: nkfreeman
"""

def data_prep(data):
    import pandas  as pd
    
    data = pd.concat([data, pd.get_dummies(data['department'])], axis=1)
    data = pd.concat([data, pd.get_dummies(data['salary_level'])], axis=1)
    
    data.columns = ['satisfaction_level', 'last_evaluation', 'number_of_projects',
                    'average_monthly_hours', 'years_with_company', 'involved_in_accident',
                    'left_company', 'promoted_last_5_years', 'department', 'salary_level',
                    'IT', 'RandD', 'accounting', 'hr', 'management', 'marketing',
                    'product_mng', 'sales', 'support', 'technical', 
                    'salary_high', 'salary_low', 'salary_medium']
    
    return data

def plot_for_interactive_vis(data, attribute):
    import pandas  as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    data_copy = data.copy()
    data_copy['satisfaction_level'] = pd.cut(data_copy['satisfaction_level'], [i*10 for i in range(11)])
    data_copy['last_evaluation'] = pd.cut(data_copy['last_evaluation'], [i*10 for i in range(11)])
    data_copy['average_monthly_hours'] = pd.cut(data_copy['average_monthly_hours'], [i*25 for i in range(3,14)])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.countplot(x = attribute, 
                  data = data_copy, 
                  hue = 'left_company', 
                  ax = ax,
                  edgecolor = 'k')
    plt.xticks(rotation = 45, fontsize = 14)
    plt.yticks(fontsize = 14)
    ax.set_xlabel(attribute, fontsize = 16)
    ax.set_ylabel('Count', fontsize = 16)
    plt.legend(title ='left_company',fontsize = 16)
    
    plt.show()
    

    
def fit_models(X_train, y_train, X_test, y_test):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    
    model_names = ['Logistic Regression',
                   'Decision Tree', 
                   'Random Forest']
    
    model_definitions = [LogisticRegression(solver = 'newton-cg'),
                         tree.DecisionTreeClassifier(),
                         RandomForestClassifier(n_estimators = 100, random_state=0)]
    
    scores = []
    best_model = None
    best_score = 0
    
    for i in range(len(model_names)):
        clf = model_definitions[i]
        clf = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        scores.append(np.round(100*clf.score(X_test, y_test),2))
        if(clf.score(X_test, y_test) > best_score):
            best_score = clf.score(X_test, y_test)
            best_model_name = model_names[i]
            best_model = clf.fit(X_train, y_train)
            
    x_vals = [i for i in range(len(model_names))]
    
    fig, ax = plt.subplots(figsize = (10,6))
    
    ax.bar(x_vals, scores, edgecolor = 'k')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(model_names, ha='right') 
        
    plt.xticks(rotation = 45, fontsize = 14)
    plt.yticks(fontsize = 14)
    ax.set_xlabel('Model', fontsize = 16)
    ax.set_ylim(0,110)
    ax.set_ylabel('Accuracy', fontsize = 16)
    ax.set_title('Model Accuracies', fontsize = 16)
    for i in range(len(scores)):
        ax.annotate(str(scores[i])+'%', 
                    xy = (x_vals[i]-0.1,scores[i]+2.5),
                    fontsize = 16)
    
    plt.show()
    
    return best_model_name, best_model, predictions


def plot_confusion_matrix(cm):
    
    import matplotlib.pyplot as plt
    import seaborn as sns    
    
    f, ax = plt.subplots(figsize=(8, 8))

    ax = sns.heatmap(cm, annot=True,
                     fmt=".0f", 
                     square = True,
                     linecolor='k',
                     annot_kws={"size": 20})
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    plt.xticks(fontsize = 18)    
    plt.yticks(fontsize = 18) 
    plt.ylabel('left_job actual', fontsize = 18);
    plt.xlabel('left_job predicted', fontsize = 18);
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize = 14)
    plt.show()