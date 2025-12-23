import os
import cv2
import time

from retinaface import RetinaFace

# Retrieve the main path
dir_path = os.path.dirname(os.path.realpath(__name__))

# Start looping through the images
begin = time.time()
count = 0
pictures = os.listdir(os.path.join(dir_path, "working"))

# Loop
for img in pictures:
    
    # Get the paht to the picture and ask RetinaFace
    imgRead = os.path.join("working", img)
    workingImg = cv2.imread(imgRead)
    resp = RetinaFace.detect_faces(workingImg)
    
    # Send to the 
    print(f"Processed picture: {count}/{len(pictures)}, time: {time.time()-begin:.6f}s")
    count += 1
    
    
    try:
        # If no faces was detected, copy-paste the image as-is.
        if resp == {}:
            cv2.imwrite(f"{os.path.join("working_faces", img)}", workingImg)
            continue
        
        # If a face is detected, extract it.
        x1, y1, x2, y2 = resp["face_1"]["facial_area"]
        subImg = workingImg[y1:y2, x1:x2] # Be cautious in RetinaFace coordinates seems to be reversed
                    
        # Store the picture with the right name 
        subImgName = f"{img.split("-")[0]}-{img.split("-")[1]}-nn-bb-{x1}-{y1}-{x2}-{y2}"
        cv2.imwrite(f"{os.path.join("working_faces", subImgName)}.jpg", subImg)
    except Exception as e:
        with open("log.txt", "a") as file:
            file.write(f"Error {e}... picture no {count}, path: {img}, {resp}.\n\n\n")