import os
import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Path to the input and output folders
input_folder = "input"
output_folder = "output"

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check if the file is an image
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=2, minNeighbors=3)#scaleFactor=1.5, minNeighbors=2

        # Crop and save only the detected faces
        for i, (x, y, w, h) in enumerate(faces):
            print(x,y,w,h)
            h2 = int(h*1.3) # ubah di perbandingan di sini secara persentase
            w2 = int(w*1.3)
            h3 = int(h*0.5)
            w3 = int(h*0.2)
            face_crop = img[y-h3:y+h2, x-w3:x+w2]
            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_face_{i}.jpg")
            cv2.imwrite(output_path, face_crop)

# Display and close windows (optional)
cv2.waitKey(0)
cv2.destroyAllWindows()
