import cntk
import numpy
from numpy import float32

training_set = numpy.array([[0,0,1,0],[0,1,0,1],[1,0,0,1],[1,1,1,0]],dtype=float32)

x = cntk.input_variable(2)
y = cntk.input_variable(2)

def getNetwork(_x):
    with cntk.layers.default_options(init=cntk.layers.glorot_uniform(), activation=cntk.relu):
        res = _x
        res = cntk.layers.Dense(4, name="l1")(res)
        res = cntk.layers.Dense(4, name="l2")(res)
        res = cntk.layers.Dense(2, name="lo", activation=None)(res)
        return res

fnn = getNetwork(x)
loss = cntk.cross_entropy_with_softmax(fnn, y)
errs = cntk.classification_error(fnn, y)
trainer = cntk.Trainer(fnn, (loss, errs), [cntk.sgd(fnn.parameters, cntk.learning_rate_schedule(0.03, cntk.UnitType.minibatch))])

for times in range(1000):
    for data in training_set:
        batch = {x: numpy.array(data[:2],dtype=float32).reshape(2), y:numpy.array(data[2:],dtype = float32).reshape(2)}
        trainer.train_minibatch(batch)
        print("\r"+str(times), end="")
print("")

#print(fnn.lo.b.value)

out = cntk.softmax(fnn)
print(numpy.argmax(out.eval({x: numpy.array([[0,0]],dtype=float32).reshape(2)})))
print(numpy.argmax(out.eval({x: numpy.array([[0,1]],dtype=float32).reshape(2)})))
print(numpy.argmax(out.eval({x: numpy.array([[1,0]],dtype=float32).reshape(2)})))
print(numpy.argmax(out.eval({x: numpy.array([[1,1]],dtype=float32).reshape(2)})))
