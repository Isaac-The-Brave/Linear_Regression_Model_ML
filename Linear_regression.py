import pandas
import numpy
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle


##Set up the data file and the parameters you want to use
File = pandas.read_csv("student-mat.csv",sep=";")
Data = File[["G1", "G2", "G3", "studytime", "absences", "failures"]]

##Set variable you wish to predict
Label = "G3"

##Set up two tables, one table with features and one with labels
#Returns a dataframe without G3
X = numpy.array(Data.drop([Label], 1))
#Set the dataframe you wish to test against
y = numpy.array(Data[Label])
## Take all the attributes and labels and split them into four arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#Set best variable as anything for now
best = 0

#Train the data-set 30 times
for _ in range(30):
    #TRAINING YOUR DATASETS, ONLY UNCOMMENT IF THE DATA HAS BEEN GENERATED INTO A .PICKLE FILE.
    ## Split all of them up to four different variables
    ## Take all the attributes and labels and split them into four arrays
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    ## Setting up the linear regression model
    linearmodel = linear_model.LinearRegression()
    linearmodel.fit(x_train, y_train)
    AccuracyRate = linearmodel.score(x_test, y_test)
    print(AccuracyRate)
    # Create Matplotlib of training
    if AccuracyRate > best:
        best = AccuracyRate
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linearmodel, f)


## Set up the PREDICTION model
# Open Matplotlib file instead of train
pickle_in = open("studentmodel.pickle", "rb")
linearmodel = pickle.load(pickle_in)
predictions = linearmodel.predict(x_test)

#Calculate the Accuracy of your results
AccuracyRate = linearmodel.score(x_test, y_test)

#Predict and print out
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print('Calculated with the following Accuracy Rate:', AccuracyRate)

#Plot the whole thing onto Matplotlib on a Scatterplot
XValue = 'G3'
style.use("ggplot")
pyplot.scatter(Data[XValue],Data[Label])
pyplot.xlabel(XValue)
pyplot.ylabel("Final Grade, with accuracy rate of:" + str(AccuracyRate))
pyplot.show()
