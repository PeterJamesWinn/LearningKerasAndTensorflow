
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import logging
#logger = tf.get_logger()
#logger.setLevel(logging.ERROR)

#tf.logging.set_verbosity(tf.loggin.Error)
#

#learning a linear function - no activation functions. Demonstrates that network can learn simple numeriacal functons.

def GenerateTrainingData():
    DesignMatrixString=[]
    TrainingValues=[]
    for data in range(1,11):
        DesignMatrixString.append(data)
    DesignMatrix=np.asarray(DesignMatrixString)
    TrainingValues=ModelFunction(DesignMatrix)
    return(DesignMatrix,TrainingValues)  

def ModelFunction(DesignMatrix):
    return(5.0+DesignMatrix*3.0)
    

def Network_1(DesignMatrix,TrainingValues):
    # Feature vector will be a single value
    # units defines number of neurons in the input layer. 
    # Input_shape indicates that each datapoint in the feature vector has only one descriptor. 
    # By default a bias is included. No regularization.
    l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    network1 = tf.keras.Sequential([l0])
    network1.compile(loss='mean_squared_error',  network1_training, network1 =Network_1(DesignMatrix,TrainingValues)


plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(network1.history['loss'])
plt.show()

print("Prediction at 50: {}".format(network1.predict([50])))
print("Weights: {}".format(l0.get_weights()))