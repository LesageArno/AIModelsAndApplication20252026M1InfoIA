import os
import time
import cv2

from ultralytics import YOLO

BATCH_SIZE = 15

def extractImagesPath() -> list[list[str], list[str]]:
    
    # Retrieve path to the file directory (absolute).
    dir_path = os.path.dirname(os.path.realpath(__name__))
    
    # picture path list to return at the end
    picturePathList = [[], []]

    # For each subfolder in src_img, retrieve the subfolder path
    for path, *_ in os.walk(os.path.join(dir_path, "src_img")):
        if path.endswith("src_img"):
            continue
        
        # For each picture in each subfolder, get the path. 
        for picture in os.listdir(path):
            picturePathList[0].append(os.path.join(path, picture))
            picturePathList[1].append(picture)

    # Return the list of path to each picture
    return picturePathList    
    
    

def extractBoundingBox(imgPaths:list[list[str], list[str]]) -> None:
    # Load a model
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
    
    # Count is here to manage the batch size
    count = 0
    imgPathsLenght = len(imgPaths[0])
    begin = time.time() # Inform user of the progress
    
    # For all the dataset, perform the operation
    while count <= imgPathsLenght - 1:
        # Run batched inference on a list of images
        results = model(imgPaths[0][count:count+BATCH_SIZE])  # return a list of Results objects
        imgName = imgPaths[1][count:count+BATCH_SIZE]
        
        # Process results list
        for i, result in enumerate(results):
            # For each detected box
            for box in result.boxes:
                
                # Get the coordinates of the box and the class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                modelClass = model.names[class_id]
                
                # Filter for person only (we do not want to have an orange or a donought...)
                if modelClass != "person":
                    continue
                
                # Extract the subimage using box coordinates and cv2
                #https://stackoverflow.com/questions/17566752/how-to-find-subimage-using-the-pil-library
                img = cv2.imread(imgPaths[0][i + count])
                subImg = img[x1:x2, y1:y2]
                
                # Store the picture with the right name 
                subImgName = f"{imgName[i].removesuffix(".jpg")}-{modelClass}-nn-bb-{x1}-{y1}-{x2}-{y2}"
                cv2.imwrite(f"{os.path.join("working", subImgName)}.jpg", subImg)
                #print(subImgName)
        
        # Update count
        count += BATCH_SIZE
        print(f"Processed picture: {count}/{imgPathsLenght}, time: {time.time()-begin:.6f}s")


if __name__ == "__main__":
    imgPaths = extractImagesPath()
    extractBoundingBox(imgPaths=imgPaths)
    