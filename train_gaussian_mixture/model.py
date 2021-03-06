# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from began import BEGAN, Config
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization
from sequential.functions import Activation, gaussian_noise

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
	config = Config()
	config.gamma = 0.5
	config.num_mixture = args.num_mixture
	config.ndim_z = 256
	config.ndim_h = 128
	config.weight_std = 0.1
	config.weight_initializer = "Normal"
	config.nonlinearity_d = "elu"
	config.nonlinearity_g = "elu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.1
	config.gradient_clipping = 1
	config.weight_decay = 0

	encoder = Sequential()
	encoder.add(gaussian_noise(std=0.1))
	encoder.add(Linear(2, 64))
	encoder.add(Activation(config.nonlinearity_d))
	# encoder.add(BatchNormalization(64))
	encoder.add(Linear(None, 64))
	encoder.add(Activation(config.nonlinearity_d))
	# encoder.add(BatchNormalization(64))
	encoder.add(Linear(None, config.ndim_h))

	decoder = Sequential()
	decoder.add(Linear(config.ndim_h, 64))
	decoder.add(Activation(config.nonlinearity_d))
	# decoder.add(BatchNormalization(64))
	decoder.add(Linear(None, 64))
	decoder.add(Activation(config.nonlinearity_d))
	# decoder.add(BatchNormalization(64))
	decoder.add(Linear(None, 2))

	generator = Sequential()
	generator.add(Linear(config.ndim_z, 128))
	generator.add(Activation(config.nonlinearity_g))
	# generator.add(BatchNormalization(128))
	generator.add(Linear(None, 128))
	generator.add(Activation(config.nonlinearity_g))
	# generator.add(BatchNormalization(128))
	generator.add(Linear(None, 2))

	params = {
		"config": config.to_dict(),
		"encoder": encoder.to_dict(),
		"decoder": decoder.to_dict(),
		"generator": generator.to_dict(),
	}

	with open(sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

began = BEGAN(params)
began.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	began.to_gpu()