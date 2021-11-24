#
#
# Define custom LSTM Cell in Keras? https://stackoverflow.com/questions/54231440/define-custom-lstm-cell-in-keras
# Creating custom RNN layers with different input and state shapes: https://github.com/tensorflow/tensorflow/issues/39626
# Hot to customize LSTM: https://github.com/keras-team/keras/issues/3329
# Attention LSTM: https://github.com/codekansas/keras-language-modeling/blob/master/attention_lstm.py
# Define LSTM Cell In Keras: https://coderedirect.com/questions/228441/define-custom-lstm-cell-in-keras
# Pytorch Timse Series Custome Cell: https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py




import keras
from keras import backend as K

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        print("=============================",type(self.kernel))
        print("=============================",self.kernel.numpy().shape)    
            
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        #print(self.recurrent_kernel)

        #self.built = True

    def call(self, inputs, states):
        #print("States",states)
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


#print(MinimalRNNCell(10).build((256, 120, 7)))