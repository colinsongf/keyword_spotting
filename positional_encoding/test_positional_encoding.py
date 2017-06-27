
import numpy as np
import tensorflow as tf

from plugins.positional_encoding import positional_encoding_op 


size = 16
length = tf.placeholder(tf.int32, [])
pe = positional_encoding_op.positional_encoding(length, size)


with tf.Session() as session:
    ret, = session.run([pe], {length: 10})
    print(ret)

