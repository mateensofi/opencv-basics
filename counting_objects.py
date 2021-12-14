# Imports
import argparse
import cv2
import imutils

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# Load the image and display it to the screen
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)

# Apply Edge Detection so that we can find the
# outline of objects in the image
edged = cv2.Canny(gray, threshold1=30, threshold2=150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# Apply Image Thresholding - Will help us remove
# lighter or darker regions and contours of images
thresh = cv2.threshold(gray, thresh=225, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# Find contours of the foreground objects in the threshold image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

# Loop over the contours
for c in cnts:
    # Draw each contour on the output image with a 3px
    # thick purple outline, the display the output
    # contours one at a time
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

# Draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# Apply erosion to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# Apply dilation to increase the size of foreground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# Masking
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked", output)
cv2.waitKey(0)
