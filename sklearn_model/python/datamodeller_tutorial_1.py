import numpy as np
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


def voter(*args):
    count1=0
    count0=0
    for i in range(0,len(args)):
        if args[i] == 0:
            count0 +=1
        else:
            count1 +=1
    if count0 > count1:
        return 0
    else:
        return 1

# Read data
train = np.genfromtxt(open("../files/train.csv",'rb'), delimiter=',')
target = np.genfromtxt(open("../files/trainLabels.csv",'rb'), delimiter=',')
test = np.genfromtxt(open("../files/test.csv",'rb'), delimiter=',')

# Scale data
train = pp.scale(train)
test = pp.scale(test)

# Select features
selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
train = selector.fit_transform(train, target)
test = selector.transform(test)

# Estimate score
classifier = SVC(C=8, gamma=0.17)
scores = cv.cross_val_score(classifier, train, target, cv=30)
print('1 Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
result = classifier.fit(train, target).predict(test)

"""
#Diff estimate
classifier2 = SVC(C=12.0,kernel='rbf',probability=True,shrinking=True);
scores2 = cv.cross_val_score(classifier2, train, target, cv=30)
print('2 Estimated score: %0.5f (+/- %0.5f)' % (scores2.mean(), scores2.std() / 2))
result2 = classifier2.fit(train, target).predict(test)

#Diff estimate
classifier3 = SVC(C=6.0,kernel='rbf',probability=True,shrinking=True);
scores3 = cv.cross_val_score(classifier3, train, target,cv=30)
print('3 Estimated score: %0.5f (+/- %0.5f)' % (scores3.mean(), scores3.std() / 2))
result3 = classifier3.fit(train, target).predict(test)

#Diff estimate
classifier4 = SVC(C=6.0,gamma=0.17,kernel='rbf',probability=True,shrinking=True);
scores4 = cv.cross_val_score(classifier4, train, target,cv=30)
print('4 Estimated score: %0.5f (+/- %0.5f)' % (scores4.mean(), scores4.std() / 2))
result4 = classifier4.fit(train, target).predict(test)"""

#Diff estimate
classifier5 = SVC(C=10.0,gamma=0.10,kernel='rbf',probability=True,shrinking=True);
scores5 = cv.cross_val_score(classifier5, train, target,cv=30)
print('5 Estimated score: %0.5f (+/- %0.5f)' % (scores5.mean(), scores5.std() / 2))
result5 = classifier5.fit(train, target).predict(test)

#Diff estimate
classifier6 = SVC(C=9.0,gamma=0.10,kernel='rbf',probability=True,shrinking=True);
scores6 = cv.cross_val_score(classifier6, train, target,cv=30)
print('6 Estimated score: %0.5f (+/- %0.5f)' % (scores6.mean(), scores6.std() / 2))
result6 = classifier6.fit(train, target).predict(test)

#Diff estimate
classifier7 = SVC(C=9.0,gamma=0.20,kernel='rbf',probability=True,shrinking=True);
scores7 = cv.cross_val_score(classifier7, train, target,cv=30)
print('7 Estimated score: %0.5f (+/- %0.5f)' % (scores7.mean(), scores7.std() / 2))
result7 = classifier7.fit(train, target).predict(test)

# Predict and save
votedResult = np.ndarray(shape=len(result));
for i in range(0,len(result)):
     votedResult[i] = voter(result5[i],result6[i])

np.savetxt('../files/tutorial_voted_1_result', votedResult, fmt='%d')    
    
np.savetxt('../files/tutorial_1_result', result6, fmt='%d')


     