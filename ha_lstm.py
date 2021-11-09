import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# =============================================================================
# The MYLSTM_SF denotes the HA-LSTM structure in the code and N=4 in this case.
# =============================================================================
    
class MYLSTM_SF(Layer):
    def __init__(self,
                 units,
                 return_sequences=True,
                 **kwargs):
        self.units = units
        self.return_sequences = return_sequences
        super(MYLSTM_SF, self).__init__(**kwargs)
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(4, input_dim, self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.recurrent_a = self.add_weight(name='recurrent_a',
                                                shape=(4, (4*self.units), self.units),
                                                initializer='orthogonal',
                                                trainable=True)        
        self.bias = self.add_weight(name='bias',
                                    shape=(4, self.units),
                                    initializer='zeros',
                                    trainable=True)     
        
        
        self.W_q = self.add_weight(name='W_q',
                                   shape=(self.units, self.units), 
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k',
                                   shape=(self.units, self.units),
                                   initializer='glorot_uniform',
                                   trainable=True)   
        self.W_v = self.add_weight(name='W_v',
                                   shape=(self.units, self.units),
                                   initializer='glorot_uniform',
                                   trainable=True) 
        self.built = True
        
            
    def step(self, inputs, states):
        x_in = inputs
        c_1 = tf.identity(states[0])
        h_1 = tf.identity(states[1])
        h_2 = tf.identity(states[2])
        h_3 = tf.identity(states[3])
        h_4 = tf.identity(states[4])       
        @tf.function
        def att_f1(x1, x2, x3, x4):
            x1 = tf.expand_dims(x1, axis=1)
            x2 = tf.expand_dims(x2, axis=1)
            x3 = tf.expand_dims(x3, axis=1)        
            x4 = tf.expand_dims(x4, axis=1)
            x_c = tf.concat([x1, x2, x3, x4], axis=1)
            return x_c        
        x_h = att_f1(h_1, h_2, h_3, h_4)
        q_h = tf.tensordot(x_h,self.W_q, axes=1)
        k_h = tf.tensordot(x_h,self.W_k, axes=1)
        v_h = tf.tensordot(x_h,self.W_v, axes=1)
        m_h = tf.matmul(q_h, tf.transpose(k_h, perm=[0,2,1]))      
        m_h = tf.divide(m_h, tf.math.pow(tf.cast(self.units, dtype=tf.float32), tf.constant([0.5])))
        m_h = tf.nn.softmax(m_h, axis=-1)
               
        e = tf.matmul(m_h, v_h)
        a_0 = tf.reshape(e, shape=[-1, 1, (4*self.units)])
        a_0 = tf.squeeze(a_0, axis=1)

        f_gate = tf.math.sigmoid(tf.nn.bias_add(tf.math.add(tf.tensordot(x_in, self.kernel[0], axes=1), tf.tensordot(a_0, self.recurrent_a[0], axes=1)), self.bias[0]))
        i_gate = tf.math.sigmoid(tf.nn.bias_add(tf.math.add(tf.tensordot(x_in, self.kernel[1], axes=1), tf.tensordot(a_0, self.recurrent_a[1], axes=1)), self.bias[1]))
        o_gate = tf.math.sigmoid(tf.nn.bias_add(tf.math.add(tf.tensordot(x_in, self.kernel[2], axes=1), tf.tensordot(a_0, self.recurrent_a[2], axes=1)), self.bias[2]))
        c_hat = tf.math.tanh(tf.nn.bias_add(tf.math.add(tf.tensordot(x_in, self.kernel[3], axes=1), tf.tensordot(a_0, self.recurrent_a[3], axes=1)), self.bias[3]))
        c_0 = tf.math.add(tf.math.multiply(f_gate, c_1), tf.math.multiply(i_gate, c_hat))
        h_out = tf.math.multiply(o_gate, tf.math.tanh(c_0))
               
        @tf.function
        def change(x1, x2, x3, x4):
            x4 = tf.identity(x3)
            x3 = tf.identity(x2)
            x2 = tf.identity(x1)
            return x2, x3, x4
        h2, h3, h4 = change(h_1, h_2, h_3, h_4)                       
        return h_out, [c_0, h_out, h2, h3, h4]
    
    def call(self, inputs):
        init1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units), name='myinitial1')
        init2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units), name='myinitial2')
        init3 = tf.zeros(shape=(tf.shape(inputs)[0], self.units), name='myinitial3')
        init4 = tf.zeros(shape=(tf.shape(inputs)[0], self.units), name='myinitial4')
        init5 = tf.zeros(shape=(tf.shape(inputs)[0], self.units), name='myinitial5')       
        outputs = K.rnn(self.step, 
                        inputs, initial_states=[init1, init2, init3, init4, init5], 
                        time_major=False)
        if self.return_sequences:
            return outputs[1]
        else:
            return outputs[0]
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units' : self.units,
            'return_sequences' : self.return_sequences
        })
        return config
