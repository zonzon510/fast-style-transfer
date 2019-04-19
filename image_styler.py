import numpy as np
import tensorflow as tf
import transform
from PIL import Image
import scipy
import scipy.misc
import time
import matplotlib.pyplot as plt


class ImageStyler():

    def __init__(self, network_path):

        self.network_path = network_path
        self.sess = tf.Session()
        self.img_placeholder = None
        self.net_output = None
        # load network
        self.load_network()

    def load_network(self):

        self.img_placeholder = tf.placeholder(tf.float32, shape=(1, 256, 256, 3),
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

    # interpolate the image to a 256 x 256 grid
    # keep aspect ratio
    plt.imshow(image)
    print(image.shape)
    height, width, _ = image.shape
    print("height: ", height)
    print("width: ", width)
    # scale the image to match the larger dimmension
    if height > width:
        # scale the height to 256
        sc_factor = 256/height
        width_new = int(width * sc_factor)

        # if the required padding is odd
        if not width_new % 2 == 0:
            width_new += 1  # make it even

        resc_im = scipy.misc.imresize(image, (256, width_new))
        pad = int((256 - width_new) / 2)  # even number to pad on both sides


        print("type(resc_im[0,0,0]): ", type(resc_im[0,0,0]))
        # print(np.array([0]).astype(np.uint8))
        resc_im_padded = np.append(resc_im, np.zeros((256, pad, 3)).astype(np.uint8), axis=1)
        resc_im_padded = np.append(np.zeros((256, pad, 3)).astype(np.uint8), resc_im_padded, axis=1)

        print("type(resc_im_padded[0,0,0]): ", type(resc_im_padded[0,0,0]))
        # exit(0)

        print("np.shape(resc_im_padded): ", np.shape(resc_im_padded))
        plt.figure(4)
        plt.imshow(resc_im_padded)
        plt.show()
        exit(0)





        plt.figure(2)
        plt.imshow(resc_im)





    plt.show()
    exit(0)




    # make sure the image is divisible by 4
    # reshaped_content_height = (image.shape[0] - image.shape[0] % 4)
    # reshaped_content_width = (image.shape[1] - image.shape[1] % 4)
    # reshaped_content_image = image[:reshaped_content_height, :reshaped_content_width, :]

    # add an axis
    reshaped_content_image = reshaped_content_image[np.newaxis, :, :, :]

    return reshaped_content_image


if __name__ == "__main__":

    img = scipy.misc.imread("puppy.jpg")
    # imagestyler = ImageStyler(network_path="pretrained-networks/dora-marr-network")
    imagestyler = ImageStyler(network_path="pretrained-networks/rain-princess-network")
    # imagestyler = ImageStyler(network_path="pretrained-networks/starry-night-network")

    time1 = time.time()
    output = imagestyler.img_transform(img)
    time2 = time.time()
    print("time elapsed: ")
    print(time2 - time1)





