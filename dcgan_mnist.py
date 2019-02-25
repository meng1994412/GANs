# import packages
from pipeline.nn.conv import DCGAN
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory")
ap.add_argument("-e", "--epochs", type = int, default = 50,
    help = "# epochs to train for ")
ap.add_argument("-b", "--batch_size", type = int, default = 128,
    help = "batch size for training")
args = vars(ap.parse_args())

# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

# load the MNIST dataset and stack the training and testing data
# points so we have additional training data
print("[INFO] loading MNIST datatset...")
((trainX, _), (testX, _)) = mnist.load_data()
trainImages = np.concatenate([trainX, testX])

# add in an extra dimension for the channel and scale the images
# into [-1, 1] (which is the range of the tanh function)
trainImages = np.expand_dims(trainImages, axis = -1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5

# build the generator
print("[INFO] building generator...")
gen = DCGAN.build_generator(7, 64, channels = 1)

# build the discriminator
print("[INFO] building discriminator...")
disc = DCGAN.build_discriminator(28, 28, 1)
discOpt = Adam(lr = 0.0002, beta_1 = 0.5, decay = 0.0002 / NUM_EPOCHS)
disc.compile(loss = "binary_crossentropy", optimizer = discOpt)

# build the adversarial model by first setting the discriminator to
# not be trainable, then combine the generator and discriminator together
print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape = (100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

# compile the GAN
ganOpt = Adam(lr = 0.0002, beta_1 = 0.5, decay = 0.0002 / NUM_EPOCHS)
gan.compile(loss = "binary_crossentropy", optimizer = discOpt)

# randomly generate some benchmark noise so we can consistently
# visualize how the generative modeling is learning
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size = (256, 100))

# loop over the epochs
for epoch in range(NUM_EPOCHS):
    # show epoch information and compute the number of batches per epoch
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

    # loop over batches
    for i in range(0, batchesPerEpoch):
        # initialize an (empty) output path
        p = None

        # select the next batch of images, then randomly generate
        # noise for the generator to predict on
        imageBatch = trainImages[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size = (BATCH_SIZE, 100))

        # generate images using the noise + generator model
        genImages = gen.predict(noise, verbose = 0)

        # concatenate the actual images and generated images
        # construct class labels for the discriminator, and shuffle the data
        X = np.concatenate((imageBatch, genImages))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        (X, y) = shuffle(X, y)

        # train the discriminator on the data
        discLoss = disc.train_on_batch(X, y)

        # train generator via the adversarial model by
        # (1) generating random noise
        # (2) training the generator with the discriminator weights frozen
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        ganLoss = gan.train_on_batch(noise, [1] * BATCH_SIZE)

        # check to see if this is the end of an epoch, and if so,
        # initialize the output path
        if i == batchesPerEpoch - 1:
            p = [args["output"], "epoch_{}_output.png".format(
                str(epoch + 1).zfill(4))]

        # otherwise, check to see if we should visualize the current
        # batch for the epoch
        else:
            # create more visualizations early in the training process
            if epoch < 10 and i % 25 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(
                str(epoch + 1).zfill(4), str(i).zfill(5))]

        # check to see if we should visualize the output of the generator
        # model on our benchmark data
        if p is not None:
            # show the loss information
            print("[INFO] Step {}_{}: discriminator_loss = {:.6f}, "
                "adversarial_loss = {:.6f}".format(epoch + 1, i, discLoss, ganLoss))

            # make predictions on the benchmark noise, scale it back
            # to the range [0, 255], and generate the montage
            images = gen.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis = -1)
            vis = build_montages(images, (28, 28), (16, 16))[0]

            # write the visualization to disk
            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)