#Isaac Palacio
import os
import gc
import sys
import cv2
import numpy as np
import tensorflow as tf
from IPython.core import ultratb
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

def representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def checkInput(W, H, dataDirectory):
    if not representsInt(W) or not representsInt(H):
        raise ValueError('Invalid option. W and H must be integers.')
    if W<1 or H<1:
        raise ValueError('Invalid option. W and H must be positive integers.')
    elif not os.path.isdir(dataDirectory):
        raise ValueError('The directory is invalid.')
    elif not os.path.isdir(dataDirectory+'/Training'):
        raise ValueError('There is not a Training folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Validaiton'):
        raise ValueError('There is not a Validaiton folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Testing'):
        raise ValueError('There is not a Testing folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Training/No_pain'):
        raise ValueError('There is not a Training/No_pain folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Training/Pain'):
        raise ValueError('There is not a Training/Pain folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Validaiton/No_pain'):
        raise ValueError('There is not a Validaiton/No_pain folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Validaiton/Pain'):
        raise ValueError('There is not a Validaiton/Pain folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Testing/No_pain'):
        raise ValueError('There is not a Testing/No_pain folder in the specify directory.')
    elif not os.path.isdir(dataDirectory+'/Testing/Pain'):
        raise ValueError('There is not a Testing/Pain folder in the specify directory.')

def getImages(path, classifier, W, H):
    images = []
    labels = []
    n = 0
    m = 0
    for filename in os.listdir(path+'/No_pain'):
        img = cv2.imread(os.path.join(path+'/No_pain',filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = classifier.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in face:
                face = img[y+((h-H)//2):y+((h+H)//2), x+((w-W)//2):x+((w+W)//2)]
            if len(face) != 0:
                images.append(np.asarray(face, dtype=np.float32)/ 255.0)
                labels.append([0])
                n = n+1
    for filename in os.listdir(path+'/Pain'):
        img = cv2.imread(os.path.join(path+'/Pain',filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = classifier.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in face:
                face = img[y+((h-H)//2):y+((h+H)//2), x+((w-W)//2):x+((w+W)//2)]
            if len(face) != 0:
                images.append(np.asarray(face, dtype=np.float32)/ 255.0)
                labels.append([1])
                m = m+1
    print('Got',n,'No_pain and',m,'Pain. A total of',n+m, 'samples.\n')
    return np.array(images), np.array(labels)

#Make error messages colorful
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

#Script Input
if len(sys.argv) != 4:
    raise ValueError('The command must follow the following structure: Project2.py W H dataDirectory.')
#Get Input
W = sys.argv[1]
H = sys.argv[2]
dataDirectory = sys.argv[3]
if representsInt(W) and representsInt(H):
    W = int(W)
    H = int(H)
else:
   raise ValueError('Invalid option. W and H must be integers.') 
checkInput(W,H,dataDirectory)

# #Manual Input
# W = 128
# H = 128
# dataDirectory = "./Project2Data"
# checkInput(W,H,dataDirectory)

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print('\nGetting Training Images...')
trainImg, trainLabels = getImages(dataDirectory+'/Training', face_cascade, W, H)
print('Getting Validation Images...')
valiImg, valiLabels = getImages(dataDirectory+'/Validaiton', face_cascade, W, H)

#Create CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(W, H, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train Model
history = model.fit(trainImg, trainLabels, epochs=5, 
                    validation_data=(valiImg, valiLabels))

#Cleaning up RAM
del trainImg, trainLabels, valiImg, valiLabels
gc.collect()

print('\nGetting Testing Images...')
testImg, testLabels = getImages(dataDirectory+'/Testing', face_cascade, W, H)

#Report
y_pred = model.predict(testImg, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print('\nReport:')
print(classification_report(testLabels, y_pred_bool))

#Test model - note, testing data is same as validation here
test_loss, test_acc = model.evaluate(testImg,  testLabels, verbose=2)
print('\nTest loss and test Accuracy:')
print(test_loss, test_acc)

print('\nConfusion Matrix:')
cm = confusion_matrix(testLabels, y_pred_bool)
print(cm)
