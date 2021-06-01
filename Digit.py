import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x,y = fetch_openml('mnist_784',version = 1,return_X_y = True)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrainscaled,ytrain)

def get_pred(image):
    imPIL = Image.open(image)
    imagebw = imPIL.convert('L')
    imagebwresize = imagebw.resize((28,28),Image.ANTIALIAS)
    imagebwinvert = PIL.ImageOps.invert(imagebwresize)
    pixelfilter = 20
    minpixel = np.percentile(imagebwinvert,pixelfilter)
    imagescaled = np.clip(imagebwinvert-minpixel,0,255)
    maxpixel = np.max(imagebwinvert)
    imagescaled = np.asarray(imagescaled)/maxpixel
    testsample = np.array(imagescaled).reshape(1,784)
    testpred =  clf.predict(testsample)
    return testpred[0]