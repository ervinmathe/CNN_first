from PIL import Image 
import numpy as np
import os

class BatchLoader:
    def __init__(self , folderPath , batchSise = 1 , imageSize = (1920 , 1080)):
        self.folder_path = folderPath
        self.batch_size = batchSise
        self.img_size = imageSize

        self.cursor = 0 
        self.allImages = sorted(os.listdir(folderPath))
        self.totalImages = len(self.allImages)

    def loadBatch(self):
        start = self.cursor
        end = self.cursor + self.batch_size if self.cursor + self.batch_size < self.allImages else self.allImages - (self.cursor + self.batch_size)
        
        batch_filenames = [self.allImages[start:end]]
        batch_images = []

        for filename in batch_filenames:
            image = Image.open(os.path.join(self.folder_path , filename))

            image = image.resize(self.img_size).convert('RGB')
            image_array = np.array(image) / 255.0
            batch_images.append(image_array)


        self.cursor = end

        if self.cursor >= self.totalImages:
            self.cursor = 0

        return np.array(batch_images)
    

    def shuffleData(self):
        return np.random.shuffle(self.allImages)
        
