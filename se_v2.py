import tensorflow as tf

class SEv2(tf.keras.Model):
    def __init__(self,out,red):
        super().__init__()
        self.out = out
        self.reduction = red
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.func1a = tf.keras.layers.Dense(self.out//self.reduction,activation='relu')
        self.func1b = tf.keras.layers.Dense(self.out//self.reduction,activation='relu')
        self.func1c = tf.keras.layers.Dense(self.out//self.reduction,activation='relu')
        self.func1d = tf.keras.layers.Dense(self.out//self.reduction,activation='relu')
        self.func2 = tf.keras.layers.Dense(self.out,activation='sigmoid')

    def call(self,x):
        inp=x
        x = self.squeeze(x)
        x1 = self.func1a(x)
        x2 = self.func1b(x)
        x3 = self.func1c(x)
        x4 = self.func1d(x)
        x = tf.concat([x1,x2,x3,x4],1)
        x = self.func2(x)
        x = tf.reshape(x,[-1,1,1,self.out])
        sc = tf.keras.layers.multiply([inp,x])
        res = tf.keras.layers.add([inp,sc])
        return res