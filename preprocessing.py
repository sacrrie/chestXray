import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

label_original=pd.read_csv("./Data_Entry_2017.csv")
label_needed=label_original[['Image Index','Finding Labels']]
label_needed.dropna(inplace=True)
#Split the string in column 2 by "|"
label_needed['Finding Labels']=label_needed['Finding Labels'].str.split("|",n=16,expand=False)
binarizer=MultiLabelBinarizer()
label_needed=label_needed.join(pd.DataFrame(binarizer.fit_transform(label_needed.pop('Finding Labels')),
                                            columns=binarizer.classes_,
                                            index=label_needed.index))
label_needed=label_needed.drop(columns='No Finding')
temp,test=train_test_split(label_needed,train_size=0.8)
train,validation=train_test_split(temp,train_size=0.875)
print("train")
print(train.head())
print("validation")
print(validation.head())
print("test")
print(test.head())
np.savetxt(r'./labels/train_list.txt', train.values,fmt="%s")
np.savetxt(r'./labels/test_list.txt', test.values,fmt="%s")
np.savetxt(r'./labels/val_list.txt', validation.values,fmt="%s")
