
from __future__ import print_function
import tensorflow as tf
import os, argparse, pathlib
import matplotlib.pyplot as plt
from data import BalanceDataGenerator
import tensorflow_addons as tfa

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=15, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='resnet50/saves', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta_train', type=str, help='Name of ckpt meta file')
parser.add_argument('--model_config', default='resnet50/saves', type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='/home/hector/COVID/data/train/train_COVIDx.txt', type=str, help='Name of train file')
parser.add_argument('--testfile', default='/home/hector/COVID/data/test/test_COVIDx.txt', type=str, help='Name of test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/hector/COVID/data', type=str, help='Path to data folder')
args = parser.parse_args()

# Parameters
learning_rate = args.lr
batch_size = args.bs
display_step = 1

# output path
outputPath = './output/'
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

with open(args.trainfile) as f:
    trainfiles = f.readlines()
with open(args.testfile) as f:
    testfiles = f.readlines()


IMG_SHAPE = (224,224,3)
train_generator = BalanceDataGenerator(trainfiles, batch_size=batch_size, datadir=args.datadir, class_weights=[1., 1., 25.], one_hot=True)
test_generator = BalanceDataGenerator(testfiles, batch_size=batch_size, is_training=False, datadir=args.datadir, class_weights=[1., 1., 25.], one_hot=True)


################################################################################################################



@tf.function
def train_step(images, labels, optimizer):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
#train_loss_array = []
#train_acc_array = []

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#test_loss_array = []
#test_acc_array = []

#################################################################################################################

base_model = tf.keras.applications.ResNet50V2(input_shape = IMG_SHAPE,
                                              include_top=False,
                                             weights='imagenet')
base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(3)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
model.load_weights('resnet50/saves/covidresnet50Weights_corrected2.h5')
####################################################################################################################

base_learning_rate = learning_rate
initial_epochs = args.epochs

loss_object = tfa.losses.SigmoidFocalCrossEntropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)

total_batch_train = len(train_generator)
total_batch_test = len(test_generator)
print('training starts...')
for epoch in range(initial_epochs):
  for i in range(total_batch_train):
    images, labels, weights = next(train_generator)
    train_step(images, labels, optimizer)

  for j in range(total_batch_test):
    test_images, test_labels, weigths = next(test_generator)
    test_step(test_images, test_labels)

  template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
  #train_loss_array.append(train_loss.result())
  #train_acc_array.append(train_accuracy.result())
  #test_loss_array.append(test_loss.result())
 # test_acc_array.append(test_accuracy.result())
  # Reinicia las metricas para el siguiente epoch.
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

# Guardar configuración JSON en el disco
json_config = model.to_json()
with open(args.model_config+'/model_config_corrected.json', 'w') as json_file:
    json_file.write(json_config)
# Guardar pesos en el disco
model.save_weights(args.weightspath+'/covidresnet50Weights_corrected2.h5')

model.save(args.model_config+'/entiremodel_corrected2.h5')
