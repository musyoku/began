# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from began import BEGAN, Config
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, Deconvolution2D, Convolution2D, PixelShuffler2D
from sequential.functions import Activation, gaussian_noise, tanh, sigmoid, reshape, reshape_1d

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

sequence_filename = args.model_dir + "/model.json"
if os.path.isfile(sequence_filename):
	print "loading", sequence_filename
	with open(sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(sequence_filename))
else:
	image_width = 96
	image_height = image_width
	ndim_z = 50
	channels = 64

	config = Config()
	config.ndim_z = ndim_z
	config.weight_std = 0.01
	config.weight_initializer = "Normal"
	config.nonlinearity = "elu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 1
	config.weight_decay = 0

	# Discriminator
	encoder = Sequential()
	encoder.add(Convolution2D(3, channels, ksize=4, stride=2, pad=1))
	# encoder.add(BatchNormalization(channels))
	encoder.add(Activation(config.nonlinearity))
	encoder.add(Convolution2D(channels, 2 * channels, ksize=4, stride=2, pad=1))
	# encoder.add(BatchNormalization(2 * channels))
	encoder.add(Activation(config.nonlinearity))
	encoder.add(Convolution2D(2 * channels, 3 * channels, ksize=4, stride=2, pad=1))
	# encoder.add(BatchNormalization(3 * channels))
	encoder.add(Activation(config.nonlinearity))
	encoder.add(Convolution2D(3 * channels, 4 * channels, ksize=4, stride=2, pad=1))
	# encoder.add(BatchNormalization(4 * channels))
	encoder.add(Activation(config.nonlinearity))
	encoder.add(Linear(None, ndim_z))

	projection_size = 6

	# Generator
	generator = Sequential()
	generator.add(Linear(ndim_z, channels * projection_size ** 2))
	# generator.add(BatchNormalization(channels * projection_size ** 2))
	generator.add(reshape((-1, channels, projection_size, projection_size)))
	generator.add(PixelShuffler2D(channels, channels, r=2))
	# generator.add(BatchNormalization(channels))
	generator.add(Activation(config.nonlinearity))
	generator.add(PixelShuffler2D(channels, channels, r=2))
	# generator.add(BatchNormalization(channels))
	generator.add(Activation(config.nonlinearity))
	generator.add(PixelShuffler2D(channels, channels, r=2))
	# generator.add(BatchNormalization(channels))
	generator.add(Activation(config.nonlinearity))
	generator.add(PixelShuffler2D(channels, 3, r=2))

	# Generator
	decoder = Sequential()
	decoder.add(Linear(ndim_z, channels * projection_size ** 2))
	# decoder.add(BatchNormalization(channels * projection_size ** 2))
	decoder.add(reshape((-1, channels, projection_size, projection_size)))
	decoder.add(PixelShuffler2D(channels, channels, r=2))
	# decoder.add(BatchNormalization(channels))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(PixelShuffler2D(channels, channels, r=2))
	# decoder.add(BatchNormalization(channels))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(PixelShuffler2D(channels, channels, r=2))
	# decoder.add(BatchNormalization(channels))
	decoder.add(Activation(config.nonlinearity))
	decoder.add(PixelShuffler2D(channels, 3, r=2))

	params = {
		"config": config.to_dict(),
		"decoder": decoder.to_dict(),
		"encoder": encoder.to_dict(),
		"generator": generator.to_dict(),
	}

	with open(sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

began = BEGAN(params)
began.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	began.to_gpu()
