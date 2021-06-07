import cv2

# Local imports
from detect_cat import detect_cat

# Load image
image = cv2.imread('pic2.jpg', 0)

# Detection
cat_detected, annotated_image = detect_cat(image)
print("Cat detected? ", cat_detected)

# UI
cv2.imshow('CatomationDetectionResult', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
