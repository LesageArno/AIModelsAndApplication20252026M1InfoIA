import os
from PIL import Image, ImageOps, ImageFilter

import time
import pandas as pd
import numpy as np

# https://github.com/elliottzheng/batch-face
from batch_face import RetinaFace as BatchRetinaFace

class RetinaExtractor():
    def __init__(self):
        self.df = None
    
    def extractImagesPath(self, src:str = "src_img") -> None:
        """Create a dataframe containing the path to each image of the dataset and the file name (stored in self.df).

        Args:
            src (str, optional): The path to the dataset. Defaults to "src_img".
        """
        # Retrieve path to the file directory (absolute).
        dir_path = os.path.dirname(os.path.realpath(__name__))
        
        # picture path list to return at the end
        picturePathList = [[], []]

        # For each subfolder in src_img, retrieve the subfolder path
        for path, *_ in os.walk(os.path.join(dir_path, src)):
            if path.endswith(src):
                continue
            
            # For each picture in each subfolder, get the path. 
            for picture in os.listdir(path):
                picturePathList[0].append(os.path.join(path, picture))
                picturePathList[1].append(picture)

        # Store the dataframe into the object
        df = pd.DataFrame(picturePathList).T
        df.columns = ["path", "file"]
        self.df = df
    
    def extractFaces(self, batch_size:int=100, threshold:float=0.95, gpu_id:int=-1, out:str = "working", gray:bool = True, additionalPreprocessing:bool = False) -> None:
        """Function to extract the face of each image whose path is stored within self.df. The extracted faces are then stored into JPEG within the `out` folder.

        Args:
            batch_size (int, optional): The size of each batch going in RetinaFace. Defaults to 100.
            threshold (float, optional): The confidence parameter of Retina. Defaults to 0.95.
            gpu_id (int, optional): Set to 0 to use GPU and to -1 to use CPU. Defaults to -1.
            out (str, optional): Folder to save all the pictures. Defaults to "working".
            gray (bool, optional): Save the picture as grayscale. Defaults to True.
            additionalPreprocessing (bool, optional): Equlize luminosity and apply gaussian filter to remove high frequencies. Defaults to False.
        """
        # Function to load batch of images on the fly
        def loadImage(path:str) -> Image:
            with Image.open(path) as im:
                return np.asarray(im)
        
        # Initialise our batched RetinaFace
        detector = BatchRetinaFace(gpu_id)
        maxCount = self.df.shape[0]
        batchNumber = (maxCount/batch_size).__ceil__()
        begin = time.time()
        undetectedFace = 0
        
        # Process the images
        for i in range(0, batchNumber):
            print(f"Batch number: {i}/{batchNumber}, processed images: {i*batch_size}/{maxCount}, undetectected faces: {undetectedFace}, {time.time()-begin:.5f}s")
            
            # Create the batch
            image_batch = self.df.iloc[i*batch_size:(i+1)*batch_size, 0].apply(loadImage).tolist()
            
            # For each face detected in the batch
            for faceIndex, face in enumerate(detector(image_batch, threshold=threshold, batch_size=batch_size, return_dict=True)):
                
                # Get rid of images with undetected faces
                if face == []:
                    undetectedFace += 1
                    continue
                
                # Extract the subimage
                x1, y1, x2, y2 = face[0]["box"].tolist()
                height, width = image_batch[faceIndex].shape[:2]
                
                # For some reason, it happen that the returned boxes are out of bound
                x1 = max(0, min(width-1, x1))
                x2 = max(0, min(width-1, x2))
                y1 = max(0, min(height-1, y1))
                y2 = max(0, min(height-1, y2))
                
                # Crop
                im = Image.fromarray(image_batch[faceIndex])
                im = im.crop((x1, y1, x2, y2))
                
                # additional Preprocessing
                if gray:
                    im = im.convert("L")
                if additionalPreprocessing:
                    im = ImageOps.equalize(im)
                    im = im.filter(ImageFilter.GaussianBlur(2))
                    
                # Save the picture
                with open(os.path.join(out, f"{self.df.iloc[i*batch_size+faceIndex, 1]}-person-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg"), "w") as file:
                    im.save(file)
    
               
if __name__ == "__main__":
    extractor = RetinaExtractor()
    extractor.extractImagesPath()
    
    # Batch number: 175/176, processed images: 17500/17534, undetectected faces: 34, 547.65211s
    extractor.extractFaces(batch_size=100)
    