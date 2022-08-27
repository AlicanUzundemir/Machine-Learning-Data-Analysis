from posixpath import split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import sys
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("Churn_Modelling.csv" , sep=";")
del df['RowNumber']
del df['Surname']
#! del df['Geography']
del df['CustomerId']
df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})
df['Cinsiyet'] = df['Gender']
del df['Gender']


#! df = pd.get_dummies(df, columns = ['Geography'])

avgs = {}
for v in df['Geography'].unique():
    avgs[ v ] = df[ df['Geography'] == v ]['Exited'].mean()

#! print(avgs)
avgs = dict(df.groupby( ['Geography'] )['Exited'].mean())
#! print(avgs)


df['Geography'] = df['Geography'].apply( lambda value: avgs[value] )

#! print(df)




#: SHUFFLE
df = df.sample(frac = 1.0)

columns = list(df.columns)
columns.remove('Exited')

tum_y = df['Exited']
tum_x = df[ columns ]


#: SPLIT INTO TRAIN TEST
number_of_rows = len(df) # 10000
train_count = int(number_of_rows * 0.60)

train = df[:train_count]
test = df[train_count:]

train_y = train['Exited']
train_x = train[ columns ]

test_y = test['Exited']
test_x = test[ columns ]


#: CREATE THE CLASSIFIERS

algorithms = [
    RandomForestClassifier(), 
    AdaBoostClassifier(), 
    GaussianNB(), 
    svm.SVC(), 
    #NearestNeighbors(), 
    MLPClassifier(random_state=1, max_iter=300) 
]

# 1600 tane 

"""
for a in algorithms:
    a.fit( train_x, train_y )
    pred = a.predict( test_x )
    print( a, accuracy_score( test_y, pred ) )
"""

"""
for o in df:
    print( o, df[o].corr( df['Exited']) )

"""



"""
for leaf_size in [5, 10, 20, 40, 50, 80, 100, 125, 175, 250]:    
    clf = RandomForestClassifier(max_depth=6, min_samples_leaf=leaf_size) # , criterion = 'entropy'
    clf.fit( train_x, train_y )
    #! pred = clf.predict( test_x )
    print("Score", clf.score( test_x, test_y ), leaf_size)
"""



for d in [2,3,4,5,6,7,8,9,10]:    
    clf = RandomForestClassifier(max_depth=d, min_samples_leaf=20) # , criterion = 'entropy'
    clf.fit( train_x, train_y )
    #! pred = clf.predict( test_x )
    print("Score", clf.score( test_x, test_y ), d)



"""
from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
for train, test in kf.split(df):
    print("%s %s" % (train, test))
    

"""




"""
clf.fit( train_x, train_y )
print( clf.predict( train_x ) )
print( clf.score( test_x, test_y ) )
"""



"""

for o in df:
    print( o, df[o].corr( df['Exited']) )




items_to_reduce = ['CreditScore', 'Tenure', 'HasCrCard', 'EstimatedSalary']
dfSUB = df[ items_to_reduce ]


pca = PCA(n_components=1)
pca_output = pca.fit_transform(dfSUB)
pca_output = pca_output[:,0]

for c in items_to_reduce: del df[c]

df['PCA_generated_feature'] = pca_output

print(df['PCA_generated_feature'].corr( df['Exited'] ))

clf = LinearDiscriminantAnalysis()
clf.fit( dfSUB, df['Exited'] )
df['LDA_generated_feature'] = clf.predict(dfSUB)
print(df)
print(df['LDA_generated_feature'].corr( df['Exited'] ))

"""





















sys.exit(1)
print("Importances", clf.feature_importances_)

for i in range(len(clf.feature_importances_)):
    print(i, train_x.columns[i], clf.feature_importances_[i], df[ train_x.columns[i] ].corr( df['Exited']) )



"""
from sklearn.tree import export_graphviz

for i in range(len(clf.estimators_)):
    estimator = clf.estimators_[i]

    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', 
                    feature_names = columns,
                    class_names = ['gidecek', 'kalacak'],
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    

    # Display in jupyter notebook
    #from IPython.display import Image
    #Image(filename = 'tree.png')
"""