import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


BATCH_SIZE = 256

def get_y(x):
	return 10 + x*x

def sample_data(n, scale=100):
	sample = []

	x = scale * (np.random.random_sample((n,)) - 0.5)

	for i in range(n):
		y = get_y(x[i])
		sample.append([x[i],y])

	return np.array(sample)

def sample_Z(size):
	return np.random.uniform(-1.,1.,size=size)

def plot_generated_data(data, iteration):
	plt.figure(figsize=(10,10))
	plt.xlabel('x')
	plt.ylabel('y')
	plt.plot(data[:,0],data[:,1], '.')
	plt.grid()
	plt.savefig('.\\results\\generated_data_{}.png'.format(iteration+1))

def plot_loss(data):
	plt.figure(figsize=(10,10))
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.plot(data.iloc[:,0],data.iloc[:,1])
	plt.plot(data.iloc[:,0],data.iloc[:,2])
	plt.legend(['Discriminator Loss','Generator Loss'])
	plt.savefig('.\\results\\loss_graph.png')


def generator():
	inputs = keras.layers.Input(shape=(2,))
	x = keras.layers.Dense(16, activation='leaky_relu')(inputs)
	x = keras.layers.Dense(16, activation='leaky_relu')(x)
	out = keras.layers.Dense(2)(x)

	model = keras.Model(inputs, out, name='Generator')

	return model

def discriminator():

	inputs = keras.layers.Input(shape=[2,])
	x = keras.layers.Dense(16,activation='leaky_relu')(inputs)
	x = keras.layers.Dense(16,activation='leaky_relu')(x)
	out = keras.layers.Dense(1, activation='sigmoid')(x)
	
	model = keras.Model(inputs, out, name='Discriminator')

	return model

adversarial_loss = keras.losses.BinaryCrossentropy()

@tf.function
def calc_gen_loss(fake_output):
	loss = adversarial_loss(tf.ones_like(fake_output), fake_output)

	return loss

@tf.function
def calc_disc_loss(real_output, fake_output):
	real_loss = adversarial_loss(tf.ones_like(real_output),real_output)
	fake_loss = adversarial_loss(tf.zeros_like(fake_output),fake_output)
	total_loss = real_loss + fake_loss

	return total_loss

generator_optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
disc_optimizer = keras.optimizers.RMSprop(learning_rate=0.001)


@tf.function
def train_step(disc, gen, Z_batch, X_batch):


	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

		generated_vector = gen(Z_batch, training=True)

		real_output = disc(X_batch, training=True)
		fake_output = disc(generated_vector, training=True)

		gen_loss = calc_gen_loss(fake_output)
		disc_loss = calc_disc_loss(real_output, fake_output)

		gradients_of_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)

		gradients_of_disc = disc_tape.gradient(disc_loss,disc.trainable_variables)

		generator_optimizer.apply_gradients(zip(gradients_of_gen,gen.trainable_variables))
		disc_optimizer.apply_gradients(zip(gradients_of_disc,disc.trainable_variables))

	return gen_loss, disc_loss

def train_gan(disc, gen):

	f = open('.\\results\\loss_logs.csv','w')
	f.write('Iteration,Discriminator Loss,Generator Loss\n')

	for i in tqdm(range(100000)):
		should_graph = ((i+1)%10000 == 0)
		save_loss = ((i+1)%10 == 0)

		X_batch = sample_data(n=BATCH_SIZE)
		Z_batch = sample_Z([BATCH_SIZE,2])

		g_loss, d_loss = train_step(disc,gen,Z_batch,X_batch)

		if save_loss:
			f.write('{},{},{}\n'.format(i,d_loss,g_loss))
		if should_graph:
			prediction = sample_Z([BATCH_SIZE,2])
			prediction = gen.predict(prediction)

			output = np.empty([BATCH_SIZE,2])

			for j, y in enumerate(prediction):
				output[j] = np.array(y)

			plot_generated_data(output, i)
	else:
		f.close()



		

disc = discriminator()
gen = generator()

train_gan(disc, gen)
loss_logs = pd.read_csv('.\\results\\loss_logs.csv')
plot_loss(loss_logs)




