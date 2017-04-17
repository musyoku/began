import sys, os
import numpy as np
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import plot
from args import args
from model import began

def plot_interpolation_generator():
	# config
	config = began.config

	num_col = 10
	num_generation = 20
	batchsize = 2 * num_generation
	base_z = began.sample_z(batchsize)

	mix_z = np.zeros((num_col * num_generation, config.ndim_z), dtype=np.float32)
	for g in xrange(num_generation):
		for i in xrange(num_col):
			mix_z[g * num_col + i] = base_z[2 * g] * (i / float(num_col)) + base_z[2 * g + 1] * (1 - i / float(num_col))

	x = began.generate_x_from_z(mix_z, test=True, as_numpy=True)
	x = (x + 1.0) / 2.0

	plot.tile_rgb_images(x.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="interpolation_generator", row=num_generation, col=num_col)

def plot_interpolation_discriminator():
	# config
	config = began.config

	num_col = 10
	num_generation = 20
	batchsize = 2 * num_generation
	base_h = np.random.uniform(-1, 1, (batchsize, config.ndim_h)).astype(np.float32)

	mix_h = np.zeros((num_col * num_generation, config.ndim_h), dtype=np.float32)
	for g in xrange(num_generation):
		for i in xrange(num_col):
			mix_h[g * num_col + i] = base_h[2 * g] * (i / float(num_col)) + base_h[2 * g + 1] * (1 - i / float(num_col))

	x = began.decode(mix_h, test=True, as_numpy=True)
	x = (x + 1.0) / 2.0
	plot.tile_rgb_images(x.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="interpolation_discriminator", row=num_generation, col=num_col)

if __name__ == '__main__':
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	plot_interpolation_generator()
	plot_interpolation_discriminator()
