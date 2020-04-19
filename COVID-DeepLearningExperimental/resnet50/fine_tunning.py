
from __future__ import print_function
import tensorflow as tf
import os, argparse, pathlib
import matplotlib.pyplot as plt
from data import BalanceDataGenerator

parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--weightspath', default='resnet50', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta_train', type=str, help='Name of ckpt meta file')
parser.add_argument('--model_config', default='resnet50/saves/COVIDresnet50_config.json', type=str, help='Name of model ckpts')
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
train_generator = BalanceDataGenerator(trainfiles, batch_size=batch_size, datadir=args.datadir, class_weights=[1., 1., 25.])
test_generator = BalanceDataGenerator(testfiles, batch_size=batch_size, is_training=False, datadir=args.datadir, class_weights=[1., 1., 25.])


################################################################################################################
with open(args.model_config) as json_file:
    json_config = json_file.read()
model = tf.keras.models.model_from_json(json_config)
model.load_weights('resnet50/saves/covidresnet50Weights_corrected.h5')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



@tf.function
def train_step(images, labels, optimizer, class_weights):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)*class_weights
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
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#train_loss_array = []
#train_acc_array = []

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#test_loss_array = []
#test_acc_array = []

model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in model.layers[:fine_tune_at - 2]:
  layer.trainable =  False

  
model.summary()
base_learning_rate = learning_rate
initial_epochs = args.epochs

fine_tune_epochs = args.epochs
total_epochs =  initial_epochs + fine_tune_epochs

optimizer2 = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10)

total_batch_train = len(train_generator)
total_batch_test = len(test_generator)

print('training starts...')
for epoch in range(fine_tune_epochs):
  for i in range(total_batch_train):
    images, labels, weights = next(train_generator)
    train_step(images, labels, optimizer2, weights)

  for j in range(total_batch_test):
    test_images, test_labels, weigths = next(test_generator)
    test_step(test_images, test_labels)

  template = 'Epoch {}, Perdida: {}, Exactitud: {}, Perdida de prueba: {}, Exactitud de prueba: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
 # train_loss_array.append(train_loss.result())
  #train_acc_array.append(train_accuracy.result())
 # test_loss_array.append(test_loss.result())
 # test_acc_array.append(test_accuracy.result())
  # Reinicia las metricas para el siguiente epoch.
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

#plot_learning_curves(train_acc_array, test_loss_array, train_loss_array, test_loss_array)
print('training has ended!')

# Guardar pesos en el disco
model.save_weights('resnet50/saves/COVIDresnet50_tuned2.h5')

model.save('resnet50/saves/entiremodel_tuned.h5')