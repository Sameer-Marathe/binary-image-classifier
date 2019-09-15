import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Flatten,Reshape
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K

'''
Matthews_correlation is often regarded as the best measure for binary classification
as latest versions of keras removed this useful metrics, I have written a version following source code of previous versions of keras
A value close to 1 is often regarded as best
'''
def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())

'''
F1 score is often a good measure to gauge a model performance
The function below calculates precision and recall and then calculates
F1 score using the formula mentioned in the return method
'''
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## initialise array to label the class images ##

y=[]

mi_array= np.load('class_a.npy')
my_array= np.load('class_b.npy')

## Labelling is done in the following lines of code, Manual one hot encode ##
for idx, el in enumerate(mi_array):
    
    y.append(np.array([1,0]))
for idx, el in enumerate(my_array):
    
    y.append(np.array([0,1]))

## Transpose the y matrix to make it a column matrix and concatenate both arrays to effectively generate our labelled dataset ##
y = np.asarray(y) 
X = np.concatenate((mi_array,my_array))

## get input shape of images ##
input_shape=X.shape[1:]

## Split the dataset to train and test dataset ##

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

##print(input_shape)

## Here we define our model
model= Sequential()
## Reshaping to cater the grayscale images for next stages ##
model.add(Reshape((40, 60, 1),input_shape=input_shape))
## convolution for images ##
model.add(Conv2D(filters=8,kernel_size = (2,2),data_format='channels_last'))
## Max Pooling to downsample the features ##
model.add(MaxPooling2D(pool_size=(2,2)))
## flatten for next dense stages ##
model.add(Flatten())

model.add(Dense(16,activation='relu'))
## dropout to ensure performance ##
model.add(Dropout(0.3))

model.add(Dense(8,activation='relu'))
## Final Sigmoid output effective for classification ##

model.add(Dense(2,activation='sigmoid'))

model.summary() ## optional ##

## here we define performance metrics ##
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[f1,'accuracy',mcor])

## fit the model ##
his=model.fit(x=X_train,y=y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test),shuffle=True)

## predict on field.npy dataset ##
y=model.predict(np.load('field.npy'))

## final performance metrics of our model ##
print('Final_Accuracy: ',  his.history.get('acc')[-1])
print('Final_Loss: ' , his.history.get('loss')[-1])
print('Final_F1: ' , his.history.get('f1')[-1])
print('Final Matthews_Correlation: ',his.history.get('mcor')[-1])

## Save the Model ## 
model.save('../model/connected_robotics.h5')
del model


