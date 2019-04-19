import numpy as np
import tensorflow as tf
import transform
from PIL import Image
import scipy
import scipy.misc
import time


class ImageStyler():

    def __init__(self, network_path, content):

        self.reshaped_content_image = reshape_image(content)
        self.network_path = network_path
        self.sess = tf.Session()

        # load network
        self.img_placeholder = None
        self.net_output = None
        self.load_network()

    def load_network(self):

        self.img_placeholder = tf.placeholder(tf.float32, shape=self.reshaped_content_image.shape,
                                         name='img_placeholder')

        self.net_output = transform.net(self.img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(self.network_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")

    def img_transform(self, image):

        reshaped_image = reshape_image(image)
        output = self.sess.run(self.net_output, feed_dict={self.img_placeholder: reshaped_image})
        im = Image.fromarray(output[0].astype(np.uint8))
        im.show()
        return output[0]


def reshape_image(image):
    reshaped_content_height = (image.shape[0] - image.shape[0] % 4)
    reshaped_content_width = (image.shape[1] - image.shape[1] % 4)
    reshaped_content_image = image[:reshaped_content_height, :reshaped_content_width, :]
    reshaped_content_image = np.ndarray.reshape(reshaped_content_image, (1,) + reshaped_content_image.shape)
    return reshaped_content_image


if __name__ == "__main__":

    img = scipy.misc.imread("puppy.jpg")
    # imagestyler = ImageStyler(network_path="pretrained-networks/dora-marr-network", content=img)
    imagestyler = ImageStyler(network_path="pretrained-networks/rain-princess-network", content=img)
    # imagestyler = ImageStyler(network_path="pretrained-networks/starry-night-network", content=img)

    time1 = time.time()
    output = imagestyler.img_transform(img)
    time2 = time.time()
    print("time elapsed: ")
    print(time2 - time1)





