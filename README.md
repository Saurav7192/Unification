# Unification

In this project we take datasets (probably of same schemas) and unify all datasets into single one.

Theory:
 
 One dataset is taken for training purpose and all other dataset is predicted on it.
 
 Step 1: We take the training dataset and treat all the coloumn of that dataset as independent category and mark it with respective column
 number representing it category.
 
 Step 2: Now taking One other dataset at a time we do as follows:
         
           1. take One coloum at a time and try to predict with every coloumn of training dataset
           2. the column of highest predicted values and append the column in that of training dataset.
   
Step 3: We do same with every other dataset 


Now how to do it?

step 1:Data_Reading(DataSets): # this function is used to read all datasets and stores its row column ratio for further calculation o f largest dataset (our training dataset).

step 2: largest_Dataset(Ratio)# this function is used to give largest dataset which is further use for training purpose.

step 3: Encoding_DataSet(DataSet) :# this function is for Encoding the dataset for mathematical modelling (because some of the column may have string data).

step 4: def Data_Framing(Dataset): # this function is used to frame the dataset into categorical range for training

step 5: ef Training_Preparation(train_data): # this function separtes the dependent and independent vriable foe training.

step 6: def Training_Models(X_train, X_test, Y_train, Y_test) :
    # this is used as path to training models

    # below are training models

    NaiveBayies().train(X_train,Y_train)

    LogisticRegression().train(X_train, Y_train)

    SVM().train(X_train, Y_train)

    DecissionTree().train(X_train, Y_train)

    xGBoost().train(X_train, Y_train)
    
  step 7 :def Testing(Dataset): # this function is used to test and predict the rest datasets by giving its individual coloumn as a different category
  
  step 8: def Comitte_Machine(Data_coloumn, data_set, result_dataset): # this function is used to store all outputs of different models
  
  step 9 : Unification of all datasets into One large dataset.
  
  
  
  This is not the very good way of procedding and also coading style is aslo not so good. But this is version 1
