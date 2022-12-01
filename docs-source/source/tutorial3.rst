.. _tutorial3:

Tutorial 3: Using a custom architecture
=======================================

Out of the box, Slideflow includes support for 21 model architectures in the Tensorflow backend and 17 with the PyTorch backend. In this tutorial, we will demonstrate how to train a custom model architecture (ViT) in either backend.

Custom Tensorflow model
***********************

Any Tensorflow/Keras model (:class:`tf.keras.Model`) can be trained in Slideflow by setting the ``model`` parameter of a :class:`slideflow.ModelParams` object to a function which initalizes the model.

First, define the model in a file that can be imported. In this example, we will define a vision transformer (ViT) model in a file ``vit_tensorflow.py``:

.. code-block:: python

    # From:
    # https://github.com/ashishpatel26/Vision-Transformer-Keras-Tensorflow-Pytorch-Examples/blob/main/Vision_Transformer_with_tf2.ipynb

    import math
    import six
    import tensorflow as tf
    from einops.layers.tensorflow import Rearrange


    def gelu(x):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            x: float Tensor to perform activation.
        Returns:
            `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf


    def get_activation(identifier):
        """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.
        It checks string first and if it is one of customized activation not in TF,
        the corresponding activation will be returned. For non-customized activation
        names and callable identifiers, always fallback to tf.keras.activations.get.
        Args:
            identifier: String name of the activation function or callable.
        Returns:
            A Python function corresponding to the activation function.
        """
        if isinstance(identifier, six.string_types):
            name_to_fn = {"gelu": gelu}
            identifier = str(identifier).lower()
            if identifier in name_to_fn:
                return tf.keras.activations.get(name_to_fn[identifier])
        return tf.keras.activations.get(identifier)


    class Residual(tf.keras.Model):

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def call(self, x):
            return self.fn(x) + x


    class PreNorm(tf.keras.Model):

        def __init__(self, dim, fn):
            super().__init__()
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
            self.fn = fn

        def call(self, x):
            return self.fn(self.norm(x))


    class FeedForward(tf.keras.Model):

        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    hidden_dim,
                    activation=get_activation('gelu')
                ),
                tf.keras.layers.Dense(dim)]
            )

        def call(self, x):
            return self.net(x)


    class Attention(tf.keras.Model):

        def __init__(self, dim, heads=8):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5
            self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
            self.to_out = tf.keras.layers.Dense(dim)
            self.rearrange_qkv = Rearrange(
                'b n (qkv h d) -> qkv b h n d',
                qkv=3,
                h=self.heads
            )
            self.rearrange_out = Rearrange('b h n d -> b n (h d)')

        def call(self, x):
            qkv = self.to_qkv(x)
            qkv = self.rearrange_qkv(qkv)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
            attn = tf.nn.softmax(dots, axis=-1)
            out = tf.einsum('bhij,bhjd->bhid', attn, v)
            out = self.rearrange_out(out)
            out = self.to_out(out)
            return out


    class Transformer(tf.keras.Model):

        def __init__(self, dim, depth, heads, mlp_dim):
            super().__init__()
            layers = []
            for _ in range(depth):
                layers.extend([
                    Residual(PreNorm(dim, Attention(dim, heads=heads))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
                ])
            self.net = tf.keras.Sequential(layers)

        def call(self, x):
            return self.net(x)


    class ViT(tf.keras.Model):

        def __init__(self, *, image_size, patch_size, num_classes,
                    dim, depth, heads, mlp_dim):
            super().__init__()
            if not image_size % patch_size == 0:
                raise ValueError('image dimensions must be divisible by the '
                                 'patch size')
            num_patches = (image_size // patch_size) ** 2
            self.patch_size = patch_size
            self.dim = dim
            self.pos_embedding = self.add_weight(
                "position_embeddings",
                shape=[num_patches + 1, dim],
                initializer=tf.keras.initializers.RandomNormal(),
                dtype=tf.float32
            )
            self.patch_to_embedding = tf.keras.layers.Dense(dim)
            self.cls_token = self.add_weight(
                "cls_token",
                shape=[1, 1, dim],
                initializer=tf.keras.initializers.RandomNormal(),
                dtype=tf.float32
            )
            self.rearrange = Rearrange(
                'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
                p1=self.patch_size,
                p2=self.patch_size
            )
            self.transformer = Transformer(dim, depth, heads, mlp_dim)
            self.to_cls_token = tf.identity
            self.mlp_head = tf.keras.Sequential([
                tf.keras.layers.Dense(mlp_dim, activation=get_activation('gelu')),
                tf.keras.layers.Dense(num_classes)
            ])

        @tf.function
        def call(self, img):
            shapes = tf.shape(img)
            x = self.rearrange(img)
            x = self.patch_to_embedding(x)
            cls_tokens = tf.broadcast_to(self.cls_token, (shapes[0], 1, self.dim))
            x = tf.concat((cls_tokens, x), axis=1)
            x += self.pos_embedding
            x = self.transformer(x)
            x = self.to_cls_token(x[:, 0])
            return self.mlp_head(x)

Next, define a function that accepts any combination of the keyword arguments ``input_shape``, ``include_top``, ``pooling``, and/or ``weights`` and returns an instanced model.

.. code-block:: python

    from vit_tensorflow import ViT

    def vit_model(image_shape, **kwargs):
        return ViT(
            image_size=input_shape[0],
            patch_size=23,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048
        )

Then, create a :class:`slideflow.ModelParams` object with your training parameters, setting the ``model`` argument equal to the function you just defined:

.. code-block:: python

    import slideflow as sf
    from vit_tensorflow impport ViT

    def vit_model(image_shape, **kwargs):
        ...

    hp = ModelParams(
        tile_px=299,
        tile_um=302,
        batch_size=32,
        model=vit_model,
        ...
    )

You can now train the model as described in :ref:`tutorial1`.

Custom PyTorch model
********************

The process is very similar when using PyTorch. In this example, instead of defining the architecture in a separate file, we will use an implementation of ViT available via PyPI:

.. code-block::

    pip3 install vit-pytorch

Next, define a function which accepts any combination of the keyword arguments ``image_size`` and/or ``pretrained`` and returns an instanced model.

.. code-block:: python

    import slideflow as sf
    from vit_pytorch impport ViT

    def vit_model(image_shape, **kwargs):
        model = ViT(
            image_size=image_size,
            patch_size=23,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        model.out_features = 1000
        return model

Finally, set the ``model`` argument of a :class:`slideflow.ModelParams` object equal to this function:

.. code-block:: python

    import slideflow as sf
    from vit_pytorch impport ViT

    def vit_model(image_shape, **kwargs):
        ...

    hp = ModelParams(
        tile_px=299,
        tile_um=302,
        batch_size=32,
        model=vit_model,
        ...
    )

You can now train the model as described in :ref:`tutorial1`.