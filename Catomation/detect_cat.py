import cv2

def detect_cat(image):

    # Detection
    detector = cv2.CascadeClassifier('assets/haarcascade_frontalcatface.xml')
    rects = detector.detectMultiScale(image, scaleFactor=1.3,
        minNeighbors=10, minSize=(150, 150))

    # Draw a rectangle surrounding each cat face
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return len(rects) > 0, image
