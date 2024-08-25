import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class AutoEncoder(tf.keras.Model):
    def __init__(self, enc, dec, name='autoencoder'):
      super().__init__()
      self.encoder = enc
      self.decoder = dec
      self.obj_fun = tf.keras.losses.MeanSquaredError()

      self.params = self.encoder.variables + self.decoder.variables

    def call(self, inputs):
      qz_x  = self.encoder(inputs)
      px_z  = self.decoder(qz_x.sample())
      x_hat = px_z.mean()

      self.loss = self.obj_fun(inputs,x_hat)

    @tf.function
    def train(self,x, optimizer):
        with tf.GradientTape() as tape:
            self.call(x)
        gradients = tape.gradient(self.loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return self.loss

    def reconstruct(self, inputs):
      qz_x  = self.encoder(inputs)
      px_z  = self.decoder(qz_x.sample())
      x_hat = px_z.mean()
      
      return x_hat
