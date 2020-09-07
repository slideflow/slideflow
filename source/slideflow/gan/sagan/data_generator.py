import cv2
import numpy as np

class DataGenerator():
    def __init__(self, image_paths, image_size, batch_size):
        self.image_paths = image_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.generator = self._generator()

    def read_image(self, image_path, crop=False):
        image = cv2.imread(image_path)

        if crop:
            image = self.crop_center(image)

        image = cv2.resize(image, (self.image_size, self.image_size))
        return image[:, :, ::-1].astype(np.float32) # convert to RGB and float

    def crop_center(self, image):
        h_center, w_center, shift = image.shape[0] // 2, image.shape[1] // 2, self.image_size//2
        
        return image[
            int(h_center-(shift)):int(h_center-(shift)+self.image_size), 
            int(w_center-(shift)):int(w_center-(shift)+self.image_size)
        ]

    def _generator(self):
        data = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        idxes = np.arange(len(self.image_paths))
        while True:
            np.random.shuffle(idxes)
            
            i = 0   
            while (i + 1) * self.batch_size <= len(self.image_paths):
                batch_paths = self.image_paths[i * self.batch_size:(i + 1) * self.batch_size]

                for j, path in enumerate(batch_paths):
                    img = self.read_image(path, crop=True)
                    data[j] = (img / 127.) - 1
                i += 1
                yield data.astype(np.float32)
