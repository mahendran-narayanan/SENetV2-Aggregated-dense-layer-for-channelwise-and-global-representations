import tensorflow as tf

class SE(tf.keras.Model):
    def __init__(self,out):
        super().__init__()
        self.out = out
        self.reduction = 32
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.func1 = tf.keras.layers.Dense(self.out//self.reduction,activation='relu')
        self.func2 = tf.keras.layers.Dense(self.out,activation='sigmoid')

    def call(self,x):
        inp=x
        x = self.squeeze(x)
        x = self.func1(x)
        x = self.func2(x)
        x = tf.reshape(x,[-1,1,1,self.out])
        sc = tf.keras.layers.multiply([inp,x])
        res = tf.keras.layers.add([inp,sc])
        return res