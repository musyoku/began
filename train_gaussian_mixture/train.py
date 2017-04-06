import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import sampler
from progress import Progress
from model import params, began
from args import args
from plot import plot_kde, plot_scatter

def plot_generator(epoch, progress):
	x = began.generate_x(10000, test=True)
	x.unchain_backward()
	x = began.to_numpy(x)
	try:
		plot_scatter(x, dir=args.plot_dir, filename="generator_scatter_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
		plot_kde(x, dir=args.plot_dir, filename="generator_kde_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
	except:
		pass

def plot_reconstruction(epoch, progress, x):
	z = began.encode(x, test=True)
	x = began.decode(z, test=True)
	x.unchain_backward()
	x = began.to_numpy(x)
	try:
		plot_scatter(x, dir=args.plot_dir, filename="reconstruction_scatter_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
		plot_kde(x, dir=args.plot_dir, filename="reconstruction_kde_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
	except:
		pass

def main():
	# settings
	max_epoch = 200
	num_updates_per_epoch = 500
	plot_interval = 5
	batchsize = 100
	scale = 2.0
	config = began.config

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	kt = 0
	lambda_k = 0.001 
	progress = Progress()
	plot_generator(0, progress)
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_d = 0
		sum_loss_g = 0
		sum_M = 0

		for t in xrange(num_updates_per_epoch):
			# sample data
			samples_real = sampler.gaussian_mixture_circle(batchsize, config.num_mixture, scale=scale, std=0.2)
			samples_fake = began.generate_x(batchsize)

			loss_real = began.compute_loss(samples_real)
			loss_fake = began.compute_loss(samples_fake)

			loss_d = loss_real - kt * loss_fake
			loss_g = loss_fake

			began.backprop_discriminator(loss_d)
			began.backprop_generator(loss_g)

			loss_d = float(loss_d.data)
			loss_g = float(loss_g.data)
			loss_real = float(loss_real.data)
			loss_fake = float(loss_fake.data)

			sum_loss_d += loss_d
			sum_loss_g += loss_g

			# update control parameters
			kt += lambda_k * (config.gamma * loss_real - loss_fake)
			kt = max(0, min(1, kt))
			M = loss_real + abs(config.gamma * loss_real - loss_fake)
			sum_M += M

			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		began.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss_d": sum_loss_d / num_updates_per_epoch,
			"loss_g": sum_loss_g / num_updates_per_epoch,
			"k": kt,
			"M": sum_M / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot_generator(epoch, progress)
			plot_reconstruction(epoch, progress, sampler.gaussian_mixture_circle(10000, config.num_mixture, scale=scale, std=0.2))

if __name__ == "__main__":
	main()
