# Wow! is a convient machine learning framework based on cntk
__wow_author__ = "DGideas"
import cntk
import numpy
class wow:
	def __init__(self, _input_shape, _output_shape, _number_of_layers=2):
		self.__input_shape = cntk.input_variable(_input_shape)
		self.__output_shape = cntk.input_variable(_output_shape)
		self.__neural_network = self.__getNetwork(_input_shape, _output_shape, _number_of_layers)
		self.__trainer = self.__getTrainer()
	def __getNetwork(self, _input_shape, _output_shape, _number_of_layers=2):
		with cntk.layers.default_options(init=cntk.layers.glorot_uniform(), activation=cntk.relu):
			res = self.__input_shape
			for x in range(_number_of_layers):
				res = cntk.layers.Dense(_input_shape, name="layer"+str(x))(res)
			res = cntk.layers.Dense(_output_shape, activation=None, name="layerOutput")(res)
			return res
	def __getTrainer(self, _learning_rate=0.03):
		loss = cntk.cross_entropy_with_softmax(self.__neural_network, self.__output_shape)
		errs = cntk.classification_error(self.__neural_network, self.__output_shape)
		return cntk.Trainer(self.__neural_network, (loss, errs), [cntk.sgd(self.__neural_network.parameters, cntk.learning_rate_schedule(_learning_rate, cntk.UnitType.minibatch))])
	def train(self, _training_data, _label):
		self.__trainer.train_minibatch({self.__input_shape: numpy.array(_training_data, dtype=numpy.float32), self.__output_shape: numpy.array(_label, dtype=numpy.float32)})
	def predict(self, _data):
		out = cntk.softmax(self.__neural_network)
		return out.eval({self.__input_shape: numpy.array(_data, dtype=numpy.float32)})
	def getNetwork(self):
		return self.__neural_network

if __name__ == "__main__":
	test = wow(2,2)
	training_set = ([0,0,1,0],[0,1,0,1],[1,0,0,1],[1,1,1,0])
	test_set = ([0,0],[0,1],[1,0],[1,1])
	count = 0
	total = 2000
	print(test.getNetwork().layerOutput.b.value)
	for x in range(total):
		for element in training_set:
			test.train(element[:2], element[2:])
		print("\r%.2f%%" % (count/total*100), end="")
		count += 1
	print("")
	print(test.getNetwork().layerOutput.b.value)
	for element in test_set:
		print(str(element) + "\t" + str(numpy.argmax(test.predict(element))))
