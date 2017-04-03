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

def plot_samples(epoch, progress):
	samples_fake = began.generate_x(10000, from_gaussian=True)
	samples_fake.unchain_backward()
	samples_fake = began.to_numpy(samples_fake)
	try:
		plot_scatter(samples_fake, dir=args.plot_dir, filename="scatter_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
		plot_kde(samples_fake, dir=args.plot_dir, filename="kde_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
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
	plot_samples(0, progress)
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

			began.backprop_generator(loss_d)
			began.backprop_discriminator(loss_g)

			sum_loss_d += float(loss_d.data)
			sum_loss_g += float(loss_g.data)

			loss_real = float(loss_real.data)
			loss_fake = float(loss_fake.data)
			
			gamma = loss_fake / loss_real
			kt += lambda_k * (gamma * loss_real - loss_fake)
			M = loss_real + abs(gamma * loss_real - loss_fake)
			sum_M += M

			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		began.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss_d": sum_loss_d / num_updates_per_epoch,
			"loss_g": sum_loss_g / num_updates_per_epoch,
			"M": sum_M / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot_samples(epoch, progress)

if __name__ == "__main__":
	main()
