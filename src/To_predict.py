from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import cv2 as cv
from skimage.transform import resize

## Our custom performance metrics ##
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

def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    #Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

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


## as we use custom metrics we should define them under custom_objects method ##
model=load_model('connected_robotics.h5',custom_objects={"f1": f1,"mcor":mcor})


## Load the numpy array to predict, field.npy is given as sample ##
img_array=np.load('field.npy')

myarr=[]
for idx, el in enumerate(img_array):
    ## Thresholding to reduce noise
    _,th2 = cv.threshold(img_array[idx].astype(np.uint8),0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ## Getting the important part of the image at the center for model to correctly classify it
    col_sum = np.where(np.sum(th2, axis = 0)>0)
    row_sum = np.where(np.sum(th2, axis = 1)>0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = th2[y1:y2, x1:x2]
    padded_image = add_padding(cropped_image, int(30-(cropped_image.shape[1]/2)), int(20-(cropped_image.shape[0]/2)), int(30-(cropped_image.shape[1]/2)), int(20- (cropped_image.shape[0]/2)))
    ## Minor adjustment for inconsistencies
    if padded_image.shape[0]==39:
        padded_image = add_padding(padded_image, 0,1,0,0)
    if padded_image.shape[1]==59:
        padded_image=add_padding(padded_image, 1,0,0,0)
    
    myarr.append(model.predict(np.expand_dims(padded_image, axis=0)).tolist())
    
    
print(myarr)
    
'''
checking the performance of model, needs true label values one hot encoded


Confusion_matrix = confusion_matrix(y_true, myarr1) ## confusion matrix ##

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, myarr1.ravel()) ## ROC curve ##
'''





