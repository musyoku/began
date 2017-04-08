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
	ndim_h = 1024

	config = Config()
	config.gamma = 0.5
	config.ndim_z = ndim_z
	config.ndim_h = ndim_h
	config.weight_std = 0.1
	config.weight_initializer = "Normal"
	config.nonlinearity_d = "leaky_relu"
	config.nonlinearity_g = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 1
	config.weight_decay = 0

	# Discriminator
	encoder = Sequential()
	encoder.add(Convolution2D(3, 32, ksize=4, stride=2, pad=1))
	encoder.add(BatchNormalization(32))
	encoder.add(Activation(config.nonlinearity_d))
	encoder.add(Convolution2D(32, 64, ksize=4, stride=2, pad=1))
	encoder.add(BatchNormalization(64))
	encoder.add(Activation(config.nonlinearity_d))
	encoder.add(Convolution2D(64, 128, ksize=4, stride=2, pad=1))
	encoder.add(BatchNormalization(128))
	encoder.add(Activation(config.nonlinearity_d))
	encoder.add(Convolution2D(128, 256, ksize=4, stride=2, pad=1))
	encoder.add(BatchNormalization(256))
	encoder.add(Activation(config.nonlinearity_d))
	encoder.add(Linear(None, ndim_h))z
	encoder.add(Activation(config.nonlinearity_d))

	projection_size = 6

	# Decoder
	decoder = Sequential()
	decoder.add(BatchNormalization(ndim_h))
	decoder.add(Linear(ndim_h, 512 * projection_size ** 2))
	decoder.add(Activation(config.nonlinearity_g))
	decoder.add(BatchNormalization(512 * projection_size ** 2))
	decoder.add(reshape((-1, 512, projection_size, projection_size)))
	decoder.add(PixelShuffler2D(512, 256, r=2))
	decoder.add(BatchNormalization(256))
	decoder.add(Activation(config.nonlinearity_d))
	decoder.add(PixelShuffler2D(256, 128, r=2))
	decoder.add(BatchNormalization(128))
	decoder.add(Activation(config.nonlinearity_d))
	decoder.add(PixelShuffler2D(128, 64, r=2))
	decoder.add(BatchNormalization(64))
	decoder.add(Activation(config.nonlinearity_d))
	decoder.add(PixelShuffler2D(64, 3, r=2))

	# Generator
	generator = Sequential()
	generator.add(Linear(ndim_z, 512 * projection_size ** 2))
	generator.add(Activation(config.nonlinearity_g))
	generator.add(BatchNormalization(512 * projection_size ** 2))
	generator.add(reshape((-1, 512, projection_size, projection_size)))
	generator.add(PixelShuffler2D(512, 256, r=2))
	generator.add(BatchNormalization(256))
	generator.add(Activation(config.nonlinearity_g))
	generator.add(PixelShuffler2D(256, 128, r=2))
	generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity_g))
	generator.add(PixelShuffler2D(128, 64, r=2))
	generator.add(BatchNormalization(64))
	generator.add(Activation(config.nonlinearity_g))
	generator.add(PixelShuffler2D(64, 3, r=2))

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
