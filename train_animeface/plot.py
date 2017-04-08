import sys, os, pylab
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import began
from dataset import load_rgb_images

def tile_rgb_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_generator_outputs(filename="generator"):
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	x = began.generate_x(100, test=True, as_numpy=True)
	x = (x + 1.0) / 2.0
	tile_rgb_images(x.transpose(0, 2, 3, 1), dir=args.plot_dir, filename=filename)

def plot_autoencoder_outputs(images, filename="autoencoder"):
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	x_true = sample_from_data(images, 100)
	z_true = began.encode(x_true, test=True)
	x_true = began.to_numpy(began.decode(z_true, test=True))
	x_true = (x_true + 1.0) / 2.0
	tile_rgb_images(x_true.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="{}_real".format(filename))

	x_fake = began.generate_x(100, test=True, as_numpy=True)
	z_fake = began.encode(x_fake, test=True)
	x_fake = began.to_numpy(began.decode(z_fake, test=True))
	x_fake = (x_fake + 1.0) / 2.0
	tile_rgb_images(x_fake.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="{}_gen".format(filename))

def sample_from_data(images, batchsize):
	example = images[0]
	height = example.shape[1]
	width = example.shape[2]
	x = np.empty((batchsize, 3, height, width), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=True)
	for j in range(batchsize):
		data_index = indices[j]
		x[j] = images[data_index]
	return x

def plot_original_data(filename="real_data"):
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	images = load_rgb_images(args.image_dir)
	x = sample_from_data(images, 100)
	x = (x + 1.0) / 2.0
	tile_rgb_images(x.transpose(0, 2, 3, 1), dir=args.plot_dir, filename=filename)

if __name__ == '__main__':
	plot_original_data()
	plot_generator_outputs()
