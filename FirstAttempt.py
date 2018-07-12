import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


def Data_Reading(DataSets):
  # this function is used to read all datasets and stores its row column ratio for further calculation o f largest dataset.

    j = 0
    Ratio = {}
    for i in DataSets :

        Readed = pd.read_csv(i)
        #print(Readed)

        Shape = Readed.shape
        #print(Shape)

        Ratio[j] =  Shape[0]/2 + Shape[1]
        #print(Ratio)

        j += 1

    key = largest_Dataset(Ratio)
    #print(DataSets[key])


    data_to_frame = Encoding_DataSet(DataSets[key])
    #print(data_to_frame)

    train_data = Data_Framing(data_to_frame)
    #print(train_data)

    X_train, X_test, Y_train, Y_test = Training_Preparation(train_data)
    #print(X_train, X_test, Y_train, Y_test)

    Training_Models(X_train, X_test, Y_train, Y_test)

    Unification().largest_data(DataSets[key])

    del DataSets[key]

    result_dataset = Testing(DataSets)

    result_dataset.to_csv("Final_Dataset", sep='\t', encoding='utf-8')





def largest_Dataset(Ratio):
    # this function is used to give largest dataset which is further use for training purpose

    j = 0

    Shorted_Ratio = Ratio[0]

    for value in Ratio.values() :

        if Shorted_Ratio <= value :

            Shorted_Ratio = value
            key = j

        j += 1

    #print(key , Shorted_Ratio)

    return key


def Encoding_DataSet(DataSet) :
    # this function is for Encoding the dataset for mathematical modelling

    LE = LabelEncoder()

    dataset_pd = pd.read_csv(DataSet)
    #print(dataset_pd)

    dataset_np = np.array(dataset_pd)
    #print(dataset_np)

    Shape = dataset_np.shape
    #print(Shape)

    j = 0
    k = 0
    for i in dataset_np :
        #print(i[j])

        if isinstance(i[j], str) == True :
            #print(i[j])

            k += 1

            if k <= 3:
                continue

            else:

                dataset_pd.ix[:,j:j+1] = LE.fit_transform(dataset_pd.ix[:,j:j+1])

                k = 0

                #print(dataset_pd.ix[:,j:j+1])
        j += 1
        if j == Shape[1]:
            break
    return dataset_pd


def Data_Framing(Dataset):
    # this function is used to frame the dataset into categorical range for training

    Shape = Dataset.shape
    #print(Shape)


    dfmj = np.empty(shape=0)
    dfmj2 = np.empty(shape=0)

    for i in range(0,Shape[1],2):

        df1 = pd.DataFrame(np.array(Dataset.ix[:,i: i+1]))
        dfnp = np.array(df1)
        dfmj = np.append(dfmj,values=dfnp)
        #print(dfmj)

        if i == Shape[1] :

           df2 = pd.DataFrame(np.array(Dataset.ix[:, i+1:]))
           dfnp2 = np.array(df2)
           dfmj = np.append(dfmj2, values=dfnp2)

        else:

            df2 = pd.DataFrame(np.array(Dataset.ix[:, i + 1:i+2]))
            dfnp2 = np.array(df2)
            dfmj2 = np.append(dfmj2, values=dfnp2)

    dfmj = pd.DataFrame(dfmj)
    #print(dfmj)

    dfmj2 = pd.DataFrame(dfmj2)
    #print(dfmj2)

    train_X = pd.concat([dfmj,dfmj2])
    #print(train_X)

    train_X = np.array(train_X)

    train_Y = np.empty(shape=0)
    j = 0
    k = 0
    for i in range(0,Shape[0]*Shape[1]):
        train_Y = np.append(train_Y,values=j)

        k += 1
        if k == Shape[0]:
            j += 1
            k = 0

    train_Y = pd.DataFrame(train_Y)
    #print(train_Y)

    train_Y = np.array(train_Y)

    train_data = np.concatenate([train_X, train_Y],axis=1)
    train_data = pd.DataFrame(train_data)
    #print(train_data)

    return train_data
def Training_Preparation(train_data):
    # this function separtes the dependent and independent vriable foe training

    train_data = np.array(train_data)

    np.random.shuffle(train_data)

    train_data = pd.DataFrame(train_data)
    #print(train_data)

    Shape = train_data.shape
    #print(Shape[0])

    X = train_data.ix[:,0:0]
    #print(X)

    Y = train_data.ix[:,1:]
    #print(Y)


    if Shape[0] > 150000:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=101)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=101)


    return X_train, X_test, Y_train, Y_test


def Training_Models(X_train, X_test, Y_train, Y_test) :
    # this is used as path to training models

    # below are training models

    NaiveBayies().train(X_train,Y_train)

    LogisticRegression().train(X_train, Y_train)

    SVM().train(X_train, Y_train)

    DecissionTree().train(X_train, Y_train)

    xGBoost().train(X_train, Y_train)



class NaiveBayies():

    gnb = GaussianNB()

    def train(self, X_train, Y_train):

        NaiveBayies.gnb.fit(X_train, Y_train)

    def predict1(self,X_test):

        return NaiveBayies.gnb.predict(X_test)

class LogisticRegression():

    clf = linear_model.LogisticRegression()

    def train(self, X_train, Y_train):

        LogisticRegression.clf.fit(X_train, Y_train)

    def predict1(self,X_test):

        return LogisticRegression.clf.predict(X_test)

class SVM():

    clf = SVC()

    def train(self, X_train, Y_train):

        SVM.clf.fit(X_train, Y_train)

    def predict1(self,X_test):

        return SVC.clf.predict(X_test)

class DecissionTree():

    dt = tree.DecisionTreeClassifier()

    def train(self, X_train, Y_train):

        DecissionTree.dt.fit(X_train, Y_train)

    def predict1(self,X_test):

        return DecissionTree.dt.predict(X_test)

class xGBoost():

    xGB = GradientBoostingClassifier()

    def train(self, X_train, Y_train):

        xGBoost.xGB.fit(X_train, Y_train)

    def predict1(self,X_test):

        return xGBoost.xGB.predict(X_test)




def Testing(Dataset):
# this function is used to test and predict the rest datasets by giving its individual coloumn as a different category
    #print(Dataset)

    for dataset in Dataset:
        #print(dataset)

        encoded_dataset = Encoding_DataSet(dataset)
        #print(encoded_dataset)

        Shape = encoded_dataset.shape
        #print(Shape)

        result_dataset = pd.DataFrame([])

        for i in range(0, Shape[1]):
           encoded_X = encoded_dataset.ix[:,i:i+1]
           #print(encoded_X)

           result_dataset = Comitte_Machine(encoded_X, dataset, result_dataset)

    return result_dataset




def Comitte_Machine(Data_coloumn, data_set, result_dataset):
# this function is used to store all outputs of different models

    predicted_NB = NaiveBayies().gnb.predict(Data_coloumn)


    predicted_LR = LogisticRegression().clf.predict(Data_coloumn)


    predicted_SVM = SVM().clf.predict(Data_coloumn)


    predicted_DT = DecissionTree().dt.predict(Data_coloumn)


    predicted_xGB = xGBoost().xGB.predict(Data_coloumn)

    final_yPredict = np.empty(shape=0)
    final_yPredict = np.append(final_yPredict, values=predicted_NB)
    final_yPredict = np.append(final_yPredict, values=predicted_LR)
    final_yPredict = np.append(final_yPredict, values=predicted_SVM)
    final_yPredict = np.append(final_yPredict, values=predicted_DT)
    final_yPredict = np.append(final_yPredict, values=predicted_xGB)
    #print(final_yPredict)

    final_dict = Dict_list(final_yPredict)
    #print(final_dict)

    key, value = Checking_dict(final_dict)
    key = int(key)
    #print(key, value)

    #Data_coloumn_pd = pd.DataFrame(Data_coloumn)

    result_dataset = Embedding(key,data_set,result_dataset)
    return result_dataset

def Embedding(key, dataset, result_dataset):
# this function is used give the coloumn of predicted dataset to further unification

    dataset = pd.read_csv(dataset)

    datacoloumn = dataset.ix[:,key:key+1]
    #print(datacoloumn)

    result_dataset = Unification().Unified_dataset(key,datacoloumn, result_dataset)

    return result_dataset





def Dict_list(y_data):
# this function is used to count all outputs which belongs to respective category
    #print(y_data)

    check = {}

    for i in y_data:
      if i not in check.keys():
        check[i] = 1
      else:
        check[i] += 1
    return check


def Checking_dict(final_dict):
# this function is used to find which datacolumn the predicted coloumnn belongs
    #print(final_dict)

    largest_value = 0

    for key , value in final_dict.items():

        if value >= largest_value:
            largest_key = key
            largest_value = value

    return largest_key, largest_value

class Unification():
# this function is used to unify all datasets inti single dataset
  data_set = "/"


  def largest_data(self,dataset):
      Unification.data_set = pd.read_csv(dataset)

  def Unified_dataset(self,key, data_coloumn,result_dataset):

      data_set = Unification.data_set

      temp_coloumn = pd.DataFrame.empty
      temp_coloumn = pd.concat([data_set.ix[:,key:key+1], data_coloumn], axis=0)

      result_dataset = pd.concat([result_dataset, temp_coloumn],axis=1)
      #print(result_dataset)
      return result_dataset




if __name__ == '__main__':

    datasets = ["SocialFee.csv", "Customer Social Feed.csv"]


    Data_Reading(datasets)






