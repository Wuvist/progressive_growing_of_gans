import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize TensorFlow session.
# tf.InteractiveSession()

config_proto = tf.ConfigProto()
config_proto.gpu_options.per_process_gpu_memory_fraction = 0.4
config_proto.gpu_options.allow_growth = True
session = tf.Session(config=config_proto)
session._default_session = session.as_default()
session._default_session.enforce_nesting = False
session._default_session.__enter__() # pylint: disable=no-member

celeba = "karras2018iclr-celebahq-1024x1024.pkl"
cats  = "karras2018iclr-lsun-cat-256x256.pkl"

with open(cats, 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
latents = np.random.RandomState(1).randn(1, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[0]] # hand-picked top-10
# latents = np.stack(np.random.RandomState(1).randn(Gs.input_shape[1]))

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])


def load_image( infilename ) :
    img = Image.open( infilename )
    # img.resize((256, 256), Image.ANTIALIAS)
    data = np.asarray(img, dtype="float32") / 255
    data = np.expand_dims(data, axis=0)
    data = np.transpose(data, [0, 3, 1, 2])
    return data

def save_image(fname, data) :
	images = np.clip(np.rint((data + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
	images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

	# Save images as PNG.
	Image.fromarray(images[0], 'RGB').save(fname)

img = load_image("ww256.png")
history = Gs.reverse_gan_for_etalons(latents, labels, img)
# img = img.transpose(3, 1, 2)



data = history[-1][1]

images = Gs.run(data, labels)

save_image("r1.png", images)
