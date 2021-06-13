import os
import sys
import cv2

# Local imports
from detect_cat import detect_cat

validation_dir = 'assets/cat-validation'
for f in os.listdir(validation_dir):
    path = os.path.join(validation_dir, f)

    # Load image
    image = cv2.imread(path, 0)

    # Detection
    cat_detected, annotated_image = detect_cat(image)
    print(path, "- Cat?", cat_detected)

    # UI
    if "--no-ui" not in sys.argv:
        cv2.imshow('CatomationDetectionResult', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

