# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj
	
class Params():
	def __init__(self, dict=None):
		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if hasattr(self, attr):
				setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			if hasattr(value, "to_dict"):
				dict[attr] = value.to_dict()
			else:
				dict[attr] = value
		return dict

	def dump(self):
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

class Config(Params):
	def __init__(self):
		self.ndim_z = 50
		self.weight_std = 0.001
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "adam"
		self.learning_rate = 0.0001
		self.momentum = 0.5
		self.gradient_clipping = 1
		self.weight_decay = 0

class BEGAN():
	def __init__(self, params):
		self.params = copy.deepcopy(params)
		self.config = to_object(params["config"])
		self.chain = sequential.chain.Chain(weight_initializer=self.config.weight_initializer, weight_std=self.config.weight_std)

		# add decoder
		decoder = sequential.from_dict(self.params["decoder"])
		self.chain.add_sequence_with_name(decoder, "decoder")
		self.decoder = decoder

		# add encoder
		encoder = sequential.from_dict(self.params["encoder"])
		self.chain.add_sequence_with_name(encoder, "encoder")
		self.encoder = encoder

		# add generator
		generator = sequential.from_dict(self.params["generator"])
		self.chain.add_sequence_with_name(generator, "generator")
		self.generator = generator

		# setup optimizer
		self.chain.setup_optimizers(self.config.optimizer, self.config.learning_rate, self.config.momentum)
		self._gpu = False

	def update_learning_rate(self, lr):
		self.chain.update_learning_rate(lr)

	def to_gpu(self):
		self.chain.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def sample_z(self, batchsize=1, gaussian=False):
		ndim_z = self.config.ndim_z
		if gaussian:
			# gaussian
			z = np.random.normal(0, 1, (batchsize, ndim_z)).astype(np.float32)
		else:
			# uniform
			z = np.random.uniform(-1, 1, (batchsize, ndim_z)).astype(np.float32)
		return z

	def generate_x(self, batchsize=1, test=False, as_numpy=False, from_gaussian=False):
		return self.generate_x_from_z(self.sample_z(batchsize, gaussian=from_gaussian), test=test, as_numpy=as_numpy)

	def generate_x_from_z(self, z, test=False, as_numpy=False):
		z = self.to_variable(z)
		x, _ = self.generator(z, test=test, return_activations=True)
		if as_numpy:
			return self.to_numpy(x)
		return x

	def encode(self, x, test=False):
		x = self.to_variable(x)
		return self.encoder(x, test=test)

	def decode(self, x, test=False):
		x = self.to_variable(x)
		return self.decoder(x, test=test)

	def compute_loss(self, x):
		x = self.to_variable(x)
		z = self.encode(x)
		_x = self.decode(z)
		return F.mean_absolute_error(x, _x)
		# return F.mean_squared_error(x, _x)

	def backprop(self, loss):
		self.chain.backprop(loss)

	def load(self, model_dir=None):
		if model_dir is None:
			raise Exception()
		self.chain.load(model_dir + "/model.hdf5")

	def save(self, model_dir=None):
		if model_dir is None:
			raise Exception()
		try:
			os.mkdir(model_dir)
		except:
			pass
		self.chain.save(model_dir + "/model.hdf5")
