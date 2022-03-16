%%time
# import the packages
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50  # this is the model we'll be using for object detection
from tensorflow.keras.applications.resnet50 import preprocess_input  # for preprocessing the input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression

# read the input image
img = cv2.imread('/content/img2.jpg')

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
    if w / float(W) < 0.2 or h / float(H) < 0.2:
        continue

    # Extract the Roi from image 
    roi = img[y:y + h, x:x + w]
    # Convert it to RGB format
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # Resize it to fit the input requirements of the model
    roi = cv2.resize(roi, (224, 224))

    # Further preprocessing 
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # Append it to our rois list
    rois.append(roi)

    # now let's store the box co-ordinates
    x1, y1, x2, y2 = x, y, x + w, y + h
    boxes.append((x1, y1, x2, y2))

# ------------ Model--------------- #
model = ResNet50(weights='imagenet')

# Convert ROIS list to arrays for predictions
input_array = np.array(rois)
print("Input array shape is ;" input_array.shape)
#---------- Make Predictions -------#
preds = model.predict(input_array)
preds = imagenet_utils.decode_predictions(preds, top=1)


# Filter the predictions with high confidence 
objects = {}
filter_boxes = []
proba = []
for (i, pred) in enumerate(preds):
	# Extract the predictions
	(imagenetID, label, prob) = pred[0]

	# Add the filter-> confidence value at 80%
	if prob >= 0.8:
		# Extract the bounding boxes for these predictions
		filter_boxes.append(boxes[i])
		proba.append(prob)


boxes = np.array([p[0] for p in objects[label]])
proba = np.array([p[1] for p in objects[label]])
boxes = non_max_suppression(boxes, proba)
# loop over all bounding boxes that were kept after applying
# non-maxima suppression
for (startX, startY, endX, endY) in boxes:
  # draw the bounding box and label on the image
  cv2.rectangle(clone, (startX, startY), (endX, endY),
    (0, 255, 0), 2)
  y = startY - 10 if startY - 10 > 10 else startY + 10
  cv2.putText(clone, label, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output after apply non-maxima suppression
plt.imshow(clone)
cv2.waitKey(0)