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
        output = self.sess.run(self.net_output, feed_dict={self.img_placeholder: reshaped_image["resc_im"]})
        scaled_to_original_image = scale_to_original(output, reshaped_image)

        plt.figure()
        plt.imshow(output[0].astype(np.uint8))
        plt.figure()
        plt.imshow(scaled_to_original_image)
        plt.show()

        exit(0)


        im = Image.fromarray(output[0].astype(np.uint8))
        im.show()
        return output[0]


def reshape_image(image):
    # interpolate the image to a 256 x 256 grid
    # keep aspect ratio
    height, width, _ = image.shape
    # scale the image to match the larger dimmension
    transform_info = dict()
    if height > width:
        transform_type = "pad_x"
        # scale the height to 256
        sc_factor = 256/height
        width_new = int(width * sc_factor)

        # if the required padding is odd
        if not width_new % 2 == 0:
            width_new += 1  # make it even

        # rescale
        resc_im = scipy.misc.imresize(image, (256, width_new))
        pad = int((256 - width_new) / 2)  # even number to pad on both sides
        # pad left and right
        resc_im_padded = np.append(resc_im, np.zeros((256, pad, 3), dtype=np.uint8), axis=1)
        resc_im_padded = np.append(np.zeros((256, pad, 3), dtype=np.uint8), resc_im_padded, axis=1)

        transform_info["resc_im"] = resc_im_padded[np.newaxis, :, :, :]
        transform_info["transform_type"] = transform_type
        transform_info["pad"] = pad
        transform_info["original_shape"] = image.shape
        return transform_info

    elif width > height:
        transform_type = "pad_y"
        # scale the width to 256
        sc_factor = 256/width
        height_new = int(height * sc_factor)

        # if the required padding is odd
        if not height_new % 2 == 0:
            height_new += 1  # make it even

        # rescale
        resc_im = scipy.misc.imresize(image, (height_new, 256))
        pad = int((256 - height_new) / 2) # even number to pad on top and bottom
        # pad top and bottom
        resc_im_padded = np.append(resc_im, np.zeros((pad, 256, 3), dtype=np.uint8), axis=0)
        resc_im_padded = np.append(np.zeros((pad, 256, 3), dtype=np.uint8), resc_im_padded, axis=0)

        transform_info["resc_im"] = resc_im_padded[np.newaxis, :, :, :]
        transform_info["transform_type"] = transform_type
        transform_info["pad"] = pad
        transform_info["original_shape"] = image.shape
        return transform_info

    else:
        # height == width
        transform_type = "scale"
        resc_im = scipy.misc.imresize(image, (256, 256))
        pad = 0
        transform_info["resc_im"] = resc_im[np.newaxis, :, :, :]
        transform_info["transform_type"] = transform_type
        transform_info["pad"] = pad
        transform_info["original_shape"] = image.shape
        return transform_info


def scale_to_original(styled_im, image_info):

    if image_info["transform_type"] == "pad_x":
        # remove padding from x
        image_cropped = styled_im[:,:,image_info["pad"]:-image_info["pad"],:]
        image_resized = scipy.misc.imresize(image_cropped[0], image_info["original_shape"])
        return image_resized

    elif image_info["transform_type"] == "pad_y":
        # remove padding from y
        image_cropped = styled_im[:,image_info["pad"]:-image_info["pad"],:,:]
        image_resized = scipy.misc.imresize(image_cropped[0], image_info["original_shape"])
        return image_resized

    else:
        # image is square
        image_resized = scipy.misc.imresize(styled_im[0], image_info["original_shape"])
        return image_resized


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





