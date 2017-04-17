import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from progress import Progress
from model import params, began
from args import args
from dataset import load_rgb_images
from plot import plot_generator_outputs, plot_autoencoder_outputs

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x_batch = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x_batch[j] = images[data_index]
	return x_batch

def main():
	images = load_rgb_images(args.image_dir)
	config = began.config

	# settings
	max_epoch = 1000
	batchsize = 16
	num_updates_per_epoch = int(len(images) / batchsize)
	plot_interval = 5

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	kt = 0
	lambda_k = 0.001 
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_d = 0
		sum_loss_g = 0
		sum_M = 0

		for t in xrange(num_updates_per_epoch):
			# sample data
			images_real = sample_from_data(images, batchsize)
			images_fake = began.generate_x(batchsize)

			loss_real = began.compute_loss(images_real)
			loss_fake = began.compute_loss(images_fake)

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
			plot_generator_outputs(filename="generator_epoch_{}_time_{}_min".format(epoch, progress.get_total_time()))
			plot_autoencoder_outputs(images, filename="autoencoder_epoch_{}_time_{}_min".format(epoch, progress.get_total_time()))

if __name__ == "__main__":
	main()