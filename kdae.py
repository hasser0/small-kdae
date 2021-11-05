from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

def create_autoencoder(input_dim, output_func, latent_dim, latent_func, hidden_dim, hidden_func):
    input_layer = Input(shape=(input_dim,))
    prev_layer = input_layer
    for dim, func in zip(hidden_dim, hidden_func):
        prev_layer = Dense(dim, activation=func)(prev_layer)
    prev_layer = Dense(latent_dim, activation=latent_func)(prev_layer)
    for dim, func in zip(reversed(hidden_dim), reversed(hidden_func)):
        prev_layer = Dense(dim, activation=func)(prev_layer)
    autoencoder = Dense(input_dim, activation=output_func)(prev_layer)
    return Model(input_layer, autoencoder)


class KDAE:

    def __init__(self, k, ae_config, optimizer='adam', loss='mse', batch_size=8) -> None:
        self.k = k
        self.ae_config = ae_config
        self.optimizer = optimizer
        self.loss = loss
        self.checkpoint_path = "./weights/cp.ckpt"
        self.pre_ae = create_autoencoder(**ae_config)
        self.autoencoders = []
        self.clusters = [[] for _ in range(k)]
    
    def pretraining(self, images, epochs, batch_size, freq=10):
        self.pre_ae.compile(optimizer=self.optimizer, loss=self.loss)
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                        save_weights_only=True,
                        verbose=1,
                        save_freq=freq*batch_size)
        self.pre_ae.fit(images, images, 
                        epochs=epochs,
                        callbacks=[cp_callback],
                        batch_size=batch_size)

    def initialize_autoencoders(self):
        for _ in range(self.k):
            ae = create_autoencoder()
            ae.compile(optimizer=self.optimizer, loss=self.loss)
            self.autoencoders.append(ae)
    
    def cluster(self, images):
        self.clusters = [[] for _ in range(self.k)]
        for i, image in enumerate(images):
            image_array = image.array
            min_mse = np.infty
            min_k = None
            for k, ae in enumerate(self.autoencoders):
                ae_array = ae.predict(image_array)
                mse = np.sum((ae_array - image_array)**2)
                if min_mse > mse:
                    min_mse = mse
                    min_k = k
            print(f"Image {i} assigned to {min_k}")
            self.clusters[min_k].append(image)
            image.prev_cluster = image.cluster
            image.cluster = min_k

    def fit_autoencoders(self, epochs=50):
        for ae, cluster in zip(self.autoencoders, self.clusters):
            if len(cluster) == 0:
              continue
            cl = np.vstack([image.array for image in cluster])
            ae.fit(cl, cl, epochs=epochs, verbose=0)
            
    def all_clusters_remain_same(self, images):
        for image in images:
            if not image.is_same_cluster():
                return False
        return True

    def cluster_images(self, images, epochs_per_iter = 50, stop=50):
        counter = 0
        while not self.all_clusters_remain_same(images) and counter < stop:
            print(f"Iteration: {counter}")
            self.cluster(images)
            print(f"Images clustered")
            self.fit_autoencoders(epochs_per_iter)
            print(f"Autoencoders fitted")
            counter += 1

    def plot_cluster(self, k, figsize=(16,16)):
        fig = plt.figure(figsize=figsize)
        images = [img.image for img in self.clusters[k]]
        columns = 4
        rows = 1+ len(images)//columns
        for i in range(len(images)):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(images[i])
        plt.show()
        

class Image():
    
    def __init__(self, path, shape=(128, 128)) -> None:
        self.path = path
        self.shape = shape
        self.array = cv2.imread(path, cv2.IMREAD_COLOR)
        self.array = cv2.resize(self.array, shape)
        self.array = self.array[:,:,::-1]
        self.array = self.array.flatten()
        self.array = self.array.reshape((1, -1))/255
        self.prev_cluster = -1
        self.cluster = -2
    
    def is_same_cluster(self):
        if self.prev_cluster == self.cluster:
            return True
        return False

    @property
    def rgb(self):
        return self.array.reshape((self.shape + (3,)))


if __name__ == "__main__":
    IMAGE_WIDTH = 64
    ae_config = {
        "input_dim":IMAGE_WIDTH*IMAGE_WIDTH*3,
        "output_func": "sigmoid",
        "latent_dim": 8,
        "latent_func": "sigmoid",
        "hidden_dim": [3072,768,192,48],
        "hidden_func": ["sigmoid"]*4
    }
    k_dae = KDAE(10, ae_config)
    PATH_TO_IMAGES = "/home/hasser/semestre_7/imac/kdae/simpson/"
    images = []
    for file in os.listdir(PATH_TO_IMAGES):
        img = Image(PATH_TO_IMAGES + file, shape=(IMAGE_WIDTH, IMAGE_WIDTH))
        images.append(img)
    pre_data = np.vstack([image.array for image in images])
    k_dae.pretraining(pre_data, 100, 8)
    k_dae.pre_ae.summary()
    original = images[0].array
    predicted = k_dae.pre_ae.predict(original)
    cv2.imwrite("./predicted.png", predicted.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 3))
    cv2.imwrite("./original.png", original.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 3))