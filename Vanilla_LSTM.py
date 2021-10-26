## LSTM Implementation with following
# Input Gate
# Output gate
# Forget gate
# PeepHole Connections


import keras
from keras import backend as K

class Vanilla_LSTM(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(Vanilla_LSTM, self).__init__(**kwargs)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)


        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent_constraint = constraints.get(recurrent_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        
        

    def build(self, input_shape):

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        #print(self.kernel)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        print(self.recurrent_kernel)


        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                        name='recurrent_kernel',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)






        #self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


print(MinimalRNNCell(10).build((256, 120, 7)))