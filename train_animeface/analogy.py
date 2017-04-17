import sys, os
import numpy as np
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import plot
from args import args
from model import began

def run_method_1():
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
	
	# optimize z
	# xp = began.xp
	# x_fake = began.generate_x_from_z(mix_z, test=True, as_numpy=False)
	# x_fake.unchain_backward()
	# for n in xrange(500):
	# 	discrimination_fake, _ = began.discriminate(x_fake, test=True)
	# 	opt = F.sum(discrimination_fake)
	# 	print opt.data
	# 	opt.backward()
	# 	# began.backprop_generator(-F.sum(discrimination_fake))
	# 	x_fake = began.to_variable(xp.clip(x_fake.data + x_fake.grad * 0.01, -1, 1))

	# x = began.to_numpy(x_fake)
	# x = (x + 1.0) / 2.0

	plot.tile_rgb_images(x.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="analogy_1", row=num_generation, col=num_col)

if __name__ == '__main__':
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	run_method_1()
