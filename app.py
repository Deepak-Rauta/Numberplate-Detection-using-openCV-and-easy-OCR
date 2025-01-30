from PIL import Image
import cv2
import pytesseract
import streamlit as st
import numpy as np

# pytesseract  path
pytesseract.pytesseract.tesseract_cmd='model/Tesseract-OCR/tesseract.exe'

def detect_image(uploaded_image):

    # Detect plates
    image=np.array(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    vehicle_cascade=cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')
    # Detect plates
    plates=vehicle_cascade.detectMultiScale(image,1.1,6)
    # Draw rectangle around the detected plate
    c=1
    st.write(f"Number of Number Plates detected={len(plates)}")
    for (x,y,w,h) in plates:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)   # B G  R  Red Is ON
        plate_area=image[y+7:y+h-7,x+7:x+w-7]

        # OCR on the plate region
        plate_text = pytesseract.image_to_string(plate_area, config='--psm 7').strip()
        st.image(plate_area, caption=f"Number_Plate Detected {c}", use_column_width=True)
        cv2.putText(image, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        st.write(f"Detected Number Plate {c}: **{plate_text}**")
        c+=1
    return image
    
# Streamlit App Interface
st.title("Vehicle Number Plate Detection")

# Upload image for plate detection
uploaded_image = st.file_uploader("Upload an Image in(jpg, jpeg) format", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)

    # Perform license plate detection on the uploaded image
    detected_image = detect_image(image)
    st.image(detected_image, caption=" Image with Detected number plates", use_column_width=True)

