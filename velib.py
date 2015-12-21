import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import theano.tensor as T
import time
import timeit
import json
import urllib
import cPickle
import csv
import datetime
from pprint import pprint

from dill.dill import FileNotFoundError

#path="c:/temp/"
path="/home/pi/velib"
#path="http://opendata.paris.fr/explore/dataset/stations-velib-disponibilites-en-temps-reel/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true"

# Global variables
nn_input_dim = 4 # input layer dimensionality
nn_output_dim = 4 # output layer dimensionality
nn_hdim = 1000 # hiden layer dimensionality


class data:
    def __init__(self):
        self.num_examples = 0

    Y=np.empty([0],np.int32)
    X=np.empty([0,nn_input_dim],np.float32)


class model:
    W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
    b1 = theano.shared(np.zeros(nn_hdim), name='b1')
    W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
    b2 = theano.shared(np.zeros(nn_output_dim), name='b2')

    def __init__(self,nn_input_dim,nn_hdim,nn_output_dim):
        np.random.seed(0)

        self.W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
        self.b1.set_value(np.zeros(nn_hdim))
        self.W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
        self.b2.set_value(np.zeros(nn_output_dim))

    @classmethod
    def save(cls, param):
        f = file(param, 'wb')
        cPickle.dump(cls.W1, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(cls.b1, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(cls.W2, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(cls.b2, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print "sauvegarde du model"

    def load(self, name):
        try:
            f = file(name, 'rb')
            self.W1=cPickle.load(f)
            self.b1=cPickle.load(f)
            self.W2=cPickle.load(f)
            self.b2=cPickle.load(f)
            f.close()
        except FileNotFoundError :
            print name + " introuvable"



def build_model(num_passes, print_loss,train):
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
        # This will update our parameters W2, b2, W1 and b1!
        gradient_step(train.X, train.Y)
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(train.X, train.Y))



def loadTrain(url):

    if(url.startswith("c:")):
        url="file:///"+url

    rc=data()
    response = urllib.urlopen(url)
    for row in csv.reader(response,delimiter=';'):
        if(row[0]!="number"):
            dt=datetime.datetime.strptime(row[11].split("+")[0], '%Y-%m-%dT%H:%M:%S')
            #format : numero station / jour de la semaine / mois / heure ==> resultat
            rc.X=np.append(rc.X,np.array([[float(row[0]),float(dt.weekday()),float(dt.month),float(dt.hour)]]),axis=0)
            sc="0b"+str(abs(int(row[8])>3))+str(abs(int(row[9])>6))
            rc.Y=np.append(rc.Y,np.array([int(sc,2)]),axis=0)

    rc.num_examples=len(rc.X)
    return rc


m=model(nn_input_dim,nn_hdim,nn_output_dim)
m.load(path+'model.obj')



# Gradient descent parameters (I picked these by hand)
epsilon = 0.005 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Our data vectors
X = T.matrix('X') # matrix of doubles
y = T.lvector('y') # vector of int64

# Forward propagation
# Note: We are just defining the expressions, nothing is evaluated here!
z1 = X.dot(m.W1) + m.b1
a1 = T.tanh(z1)
z2 = a1.dot(m.W2) + m.b2
y_hat = T.nnet.softmax(z2) # output probabilties
iteration=0


# Returns a class prediction
prediction = T.argmax(y_hat, axis=1)
predict = theano.function([X], prediction)


for iteration in range(0,0):
    #train = loadTrain(path+"stationvelib.csv")
    train = loadTrain("http://opendata.paris.fr/explore/dataset/stations-velib-disponibilites-en-temps-reel/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true")

    # The regularization term (optional)
    loss_reg = 1./train.num_examples * reg_lambda/2 * (T.sum(T.sqr(m.W1)) + T.sum(T.sqr(m.W2)))
    # the loss function we want to optimize
    loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg


    # Theano functions that can be called from our Python code
    forward_prop = theano.function([X], y_hat)
    calculate_loss = theano.function([X, y], loss)

    # Easy: Let Theano calculate the derivatives for us!
    dW2 = T.grad(loss, m.W2)
    db2 = T.grad(loss, m.b2)
    dW1 = T.grad(loss, m.W1)
    db1 = T.grad(loss, m.b1)

    gradient_step = theano.function([X, y],
        updates=((m.W2, m.W2 - epsilon * dW2),
                 (m.W1, m.W1 - epsilon * dW1),
                 (m.b2, m.b2 - epsilon * db2),
                 (m.b1, m.b1 - epsilon * db1)))

    build_model(1000,True,train)
    model.save(path+'model.obj')


m.load(path+'model.obj')
print predict([[10011,1,12,13]])
