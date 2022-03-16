# import the packages
import numpy as np
import cv2
from tensorflow.keras.applications import Xception  # this is the model we'll be using for object detection
from tensorflow.keras.applications.xception import preprocess_input  # for preprocessing the input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression

# read the input image
img = cv2.imread('/content/img4.jpg')

# instanciate the selective search segmentation algorithm of opencv
search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
search.setBaseImage(img)  # set the base image as the input image
search.switchToSelectiveSearchFast()  
# you can also use this for more accuracy -> 
# search.switchToSelectiveSearchQuality()

rects = search.process()  # process the image

rois = []
boxes = []
(H, W) = img.shape[:2]
for (x, y, w, h) in rects:
    # check if the ROI has atleast 
    # 20% the size of our image
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue

    # Extract the Roi from image 
    roi = img[y:y + h, x:x + w]
    # Convert it to RGB format
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # Resize it to fit the input requirements of the model
    roi = cv2.resize(roi, (299, 299))

    # Further preprocessing 
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # Append it to our rois list
    rois.append(roi)

    # now let's store the box co-ordinates
    x1, y1, x2, y2 = x, y, x + w, y + h
    boxes.append((x1, y1, x2, y2))

# ------------ Model--------------- #
model = Xception(weights='imagenet')


# Convert ROIS list to arrays for predictions
input_array = np.array(rois)
print("Input array shape is ;" ,input_array.shape)
#---------- Make Predictions -------#
preds = model.predict(input_array)
preds = imagenet_utils.decode_predictions(preds, top=1)


# Initiate the dictionary
objects={}
for (i, pred) in enumerate(preds):
  # extract the prediction tuple 
  # and store it's values
  iD = pred[0][0]
  label = pred[0][1]
  prob = pred[0][2]
    
  if prob >= 0.9:
        # grab the bounding box associated with the prediction and
        # convert the coordinates
        box = boxes[i]

    # create a tuble using box and probability
        value = objects.get(label, [])
        # append the value to the list for the label
        value.append((box, prob))
  # Add this tuple to the objects dictionary that we initiated
        objects[label] = value
print(objects)

# Loop through the labels
# for each label apply the non_max_suppression
for label in objects.keys():
    # clone the original image so that we can draw on it
    img_copy = img.copy()
    boxes = np.array([pred[0] for pred in objects[label]])
    proba = np.array([pred[1] for pred in objects[label]])
    boxes = non_max_suppression(boxes, proba)
    print(boxes)
    # Let's create the bounding box now
    (startX, startY, endX, endY) = boxes[0]

    # Draw the bounding box
    cv2.rectangle(img_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
   
    # Put the label on the image   
    cv2.putText(img_copy, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0), 2)

    cv2.imshow("Regional proposal object deteciton", img_copy)
    cv2.waitKey(0)






