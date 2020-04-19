import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from data import BalanceDataGenerator
import os, argparse, pathlib
import seaborn as sn
import pandas as pd

parser = argparse.ArgumentParser(description='COVID-Net Evaluation Script')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='resnet50/saves/covidresnet50Weights_corrected2.h5', type=str, help='Path to output folder')
parser.add_argument('--model_config', default='resnet50/saves/model_config_corrected.json', type=str, help='Name of model ckpts')
parser.add_argument('--testfile', default='/home/hector/COVID/data/test/test_COVIDx.txt', type=str, help='Name of test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/hector/COVID/data', type=str, help='Path to data folder')
args = parser.parse_args()


np.set_printoptions(precision=2)

####### LOAD MODEL AND METRICS ################################
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

with open(args.model_config) as json_file:
    json_config = json_file.read()
model = tf.keras.models.model_from_json(json_config)
model.load_weights('COVIDresnet50.h5')

def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  return predictions

######## SET IMAGE GENERATOR TO TEST SET #######################
with open(args.testfile) as f:
    testfiles = f.readlines()
print(len(testfiles))
test_generator = BalanceDataGenerator(testfiles, 
                                      batch_size=args.bs,
                                      is_training=False, 
                                      datadir=args.datadir, 
                                      class_weights=[1., 1., 25.])
total_batch_test = len(test_generator)

#print('test gen'+str(len(test_generator)))
#imgs = 0
y_pred = []
y_true = []
for j in range(total_batch_test):

    test_images, test_labels, weigths = next(test_generator)
    print(test_labels)
    
    #imgs += len(test_images)
    predictions = test_step(test_images, test_labels)
    predictions = tf.nn.softmax(predictions)
    predictions_classnum = np.argmax(predictions, axis=1)
    confidences = np.amax(predictions)

    for item in range(args.bs):
        y_pred.append(predictions_classnum[item])
        y_true.append(int(test_labels[item]))

    #print( predictions)
#print('imgs'+str(imgs))

#do the classification




target_names = ['Normal', 'Pneumonia', 'COVID-19']
print(classification_report(y_true, y_pred, target_names=target_names))


cm = confusion_matrix(y_true, y_pred)
print(cm)
#cmn = confusion_matrix(y_true, y_pred, normalize='all')


df_cm = pd.DataFrame(cm, index = ['normal','pneumonia','covid-19'],
                  columns = ['normal','pneumonia','covid-19'])

print(df_cm)

'''
#plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)



plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()


f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(image_datas[0])
axarr[0,1].imshow(image_datas[1])
axarr[1,0].imshow(image_datas[2])
axarr[1,1].imshow(image_datas[3])
'''