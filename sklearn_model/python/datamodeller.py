import numpy as np
import scipy
import sys
from sklearn import linear_model,svm,ensemble
from scipy.io import arff
import scipy.spatial.distance as spsd
from itertools import imap
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def pearsonr(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  psum = sum(imap(lambda x, y: x * y, x, y))
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den        
  
  
def doSVM(xTrain,yTrain,xTest):
    clf = svm.SVC(C=12.0,kernel='rbf',probability=True,shrinking=True);
    clf.fit(xTrain, yTrain);
    print "SVM Mean Accuracy score = ",cross_val_score(clf,xTrain,yTrain).mean();    
    yPredSVM = clf.predict(xTest);
    return clf,yPredSVM;   

def doLR(xTrain,yTrain,xTest):
    clf = linear_model.LogisticRegression();
    clf.fit(xTrain, yTrain);
    print "LR Mean Accuracy score = ",cross_val_score(clf,xTrain,yTrain).mean();
    yPredLR = clf.predict(xTest);
    return clf,yPredLR;          

def doRandomForst(xTrain,yTrain,xTest):
    clf = RandomForestClassifier();
    clf.fit(xTrain, yTrain);
    print "Random forest Accuracy score = ",cross_val_score(clf,xTrain,yTrain).mean();
    yPredLR = clf.predict(xTest);
    return clf,yPredLR;               
                           
                           
class DataModeller:
    
    def __init__(self, training_file, test_file,training_label_file):
        self.training_file = training_file
        self.test_file = test_file
        self.training_label_file=training_label_file
        
    def runAnalysis(self):
        
        trainingData = np.loadtxt(open(self.training_file, 'rb'), delimiter = ',', skiprows = 0);
        testData = np.loadtxt(open(self.test_file,'rb'), delimiter = ',', skiprows = 0);
        trainingLabel = np.loadtxt(open(self.training_label_file,'rb'), delimiter = ',', skiprows = 0);
        
        xTrain =  trainingData[:,0:]
        yTrain = trainingLabel[:]       
        xTest =  testData[:,0:]
        

        
        clf1,yPredSVM1 = doSVM(xTrain[0:300,:], yTrain[0:300], xTest);
        clf2,yPredSVM2 = doSVM(xTrain[300:600,:], yTrain[300:600], xTest);
        clf3,yPredSVM3 = doSVM(xTrain[600:1000,:], yTrain[600:1000], xTest);
        
        clflr1,yPredLR1 = doLR(xTrain[0:500,:], yTrain[0:500], xTest);

        
        clf,yPredSVM = doSVM(xTrain, yTrain, xTest);
        
        clf_rf_1,yPredRF1 = doRandomForst(xTrain, yTrain, xTest);
        
        outputFile = open("../files/final_boosted_classifiedvalues.csv", 'w+')
        """Boosting technique"""
        for i in range(0,len(xTest)):
            count1 =0
            count0 =0
            if yPredSVM1[i] == 1:
                count1+=1;
            else:
                count0+=1;
            if yPredSVM2[i] == 1:
                count1+=1;
            else:
                count0+=1;
            if yPredSVM3[i] == 1:
                count1+=1;
            else:
                count0+=1;
            if yPredLR1[i] == 1:
                count1+=1;
            else:
                count0+=1;
            if yPredSVM[i] == 1:
                count1 += 2;
            else:
                count0 += 2;
            if yPredRF1[i] == 1:
                count1 += 2;
            else:
                count0 += 2;                
                
                
                
            if count1 > count0 :
                outputFile.write(str('1')+"\n")
            else:
                outputFile.write(str('0')+"\n")
            
        outputFile.close(); 
        
        #Applying Semi supervised technique 
        """ count=0;
        for i in range(0,len(yPredSVM2)):
            if(yPredSVM1[i] == yPredSVM2[i]):
                count+=1;                
               
        xTrainNew = np.ndarray(shape=(count,xTrain.shape[1]));
        yTrainNew = np.ndarray(shape=(count));      
        xTestNew = np.ndarray(shape=(len(yPredSVM2)-count,xTrain.shape[1]));
        
        j=0;
        k=0;
        count=0;        
        for i in range(0,len(yPredSVM2)):
            if(yPredSVM1[i] == yPredSVM2[i]):
                xTrainNew[j]=xTest[i];
                yTrainNew[j]=yPredSVM1[i];
                j+=1;
                count+=1;                
            else:
                xTestNew[k]=xTest[i];
                k+=1;
            
        print count;
        
        
        xTrainNew = np.concatenate((xTrainNew,xTrain))
        yTrainNew = np.concatenate((yTrainNew,yTrain))

        clf3,yPredSVM3 = doSVM(xTrainNew, yTrainNew, xTest);
                                       
        outputFile = open("../files/final_classifiedvalues.csv", 'w+')
        for i in range(0,len(yPredSVM3)):
             outputFile.write(str(int(yPredSVM3[i]))+"\n")            
        outputFile.close(); """

 

if __name__ == '__main__':
    if len(sys.argv) < 3:       
        print 'python datamodeller.py <training-file-path>  <test-file-path>'
        sys.exit(1)
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    training_label_file = sys.argv[3]
    model = DataModeller(training_file, test_file, training_label_file)
    model.runAnalysis()
        
    
