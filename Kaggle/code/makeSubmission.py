from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create training and test sets, skipping header row [1:]
    dataset = genfromtxt(open('../datasets/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train  = [x[1:] for x in dataset]

    test = genfromtxt(open('../datasets/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train randomforest
    rf = RandomForestClassifier(n_estimators=100,n_jobs=4)
    rf.fit(train,target)

    predicted_probs = [[index+1,x[1]] for index,x in enumerate(rf.predict_proba(test))]

    savetxt('../datasets/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='MoleculeId,PredictedProbability', comments='')

if __name__ == "__main__":
    main()
