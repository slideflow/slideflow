import tensorflow as tf

'''
Xception activation layer names, for reference
	xception_activation_layer_names = [	'block1_conv1_act', 	# 32 channels
										'block1_conv2_act',		# 64 channels
										'block3_sepconv1_act', 	# 128 channels
										'block4_sepconv1_act', 	# 256 channels
										'block5_sepconv1_act', 	# 728 channels
										'block6_sepconv1_act', 	# 728 channels
										'block7_sepconv1_act', 	# 728 channels
										'block8_sepconv1_act', 	# 728 channels
										'block9_sepconv1_act', 	# 728 channels
										'block10_sepconv1_act', # 728 channels
										'block11_sepconv1_act', # 728 channels
										'block12_sepconv1_act', # 728 channels
										'block13_sepconv1_act', # 728 channels
										'block14_sepconv2_act' ]  # 2048 channels

VGG16 model, at 299x299, for reference

tile_image (InputLayer)      [(None, 299, 299, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 299, 299, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 299, 299, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 149, 149, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 149, 149, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 149, 149, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 74, 74, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 74, 74, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 74, 74, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 74, 74, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 37, 37, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 37, 37, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 37, 37, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 37, 37, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 18, 18, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
global_max_pooling2d (Global (None, 512)               0         
=================================================================
'''

def flatten_model(model_nested):
	'''Utility to flatten a nested model.'''
	def get_layers(layers):
		layers_flat = []
		for layer in layers:
			try:
				layers_flat.extend(get_layers(layer.layers))
			except AttributeError:
				layers_flat.append(layer)
		return layers_flat

	model_flat = tf.keras.models.Sequential(
		get_layers(model_nested.layers)
	)
	return model_flat