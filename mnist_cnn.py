__author__ = "DGideas"
import numpy
from numpy import float32
import matplotlib.pyplot
import cntk
import time
import sys
from ut import unittest as mnist_unittest
cntk.cntk_py.set_fixed_random_seed(1)
mnist = mnist_unittest(__author__)
training_set = numpy.array(mnist.getTrainingSet(), dtype=float32)
test_set = numpy.array(mnist.getTestSet(), dtype=float32)

x = cntk.input_variable((1,28,28))
y = cntk.input_variable(10)
def create_model(features):
	with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.relu):
		h = features
		h = cntk.layers.Convolution2D(filter_shape=(5,5), num_filters=8, strides=(2,2), pad=True, name='layer1')(h)
		h = cntk.layers.Convolution2D(filter_shape=(5,5), num_filters=16, strides=(2,2), pad=True, name='layer2')(h)
		res = cntk.layers.Dense(10, activation=None, name='layer3')(h)
		return res

cnn = create_model(x)
cnn = cnn(x/255)
loss = cntk.cross_entropy_with_softmax(cnn, y)
errs = cntk.classification_error(cnn, y)
trainer = cntk.Trainer(cnn, (loss, errs), [cntk.sgd(cnn.parameters, cntk.learning_rate_schedule(0.03, cntk.UnitType.minibatch))])
count = 0
begin_time = time.time()
for data in training_set:
	trainer.train_minibatch({x: numpy.array(data[1:], dtype=float32).reshape(1, 28, 28), y: numpy.array([1 if x==int(data[0]) else 0 for x in range(10)], dtype=float32)})
	count += 1
	print("\r%.2f%%" % (count/len(training_set)*100), file=sys.stderr, end="")
print("")
print("finish training, spent " + str(int(time.time()-begin_time)) + "s")
print(cnn.layer3.b.value)
out = cntk.softmax(cnn)
time.sleep(1)
res_mnist = []
for testcase in test_set:
	#res = out.eval(numpy.array(testcase).reshape(1, 28, 28))
	res = out.eval({x: numpy.array(testcase, dtype=float32).reshape(1, 28, 28)})
	ans = numpy.argmax(res)
	print(ans, end="")
	res_mnist.append(ans)
mnist.putAnswer(res_mnist)
