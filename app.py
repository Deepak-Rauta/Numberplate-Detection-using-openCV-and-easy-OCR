# from PIL import Image
# import cv2
# import pytesseract
# import streamlit as st
# import numpy as np

# # pytesseract  path
# pytesseract.pytesseract.tesseract_cmd='model/Tesseract-OCR/tesseract.exe'

# def detect_image(uploaded_image):

#     # Detect plates
#     image=np.array(uploaded_image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     vehicle_cascade=cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')
#     # Detect plates
#     plates=vehicle_cascade.detectMultiScale(image,1.1,6)
#     # Draw rectangle around the detected plate
#     c=1
#     st.write(f"Number of Number Plates detected={len(plates)}")
#     for (x,y,w,h) in plates:
#         cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)   # B G  R  Red Is ON
#         plate_area=image[y+7:y+h-7,x+7:x+w-7]

#         # OCR on the plate region
#         plate_text = pytesseract.image_to_string(plate_area, config='--psm 7').strip()
#         st.image(plate_area, caption=f"Number_Plate Detected {c}", use_column_width=True)
#         cv2.putText(image, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
#         st.write(f"Detected Number Plate {c}: **{plate_text}**")
#         c+=1
#     return image
    
# # Streamlit App Interface
# st.title("Vehicle Number Plate Detection")

# # Upload image for plate detection
# uploaded_image = st.file_uploader("Upload an Image in(jpg, jpeg) format", type=["jpg", "jpeg", "png"])
# if uploaded_image:
#     image = Image.open(uploaded_image)

#     # Perform license plate detection on the uploaded image
#     detected_image = detect_image(image)
#     st.image(detected_image, caption=" Image with Detected number plates", use_column_width=True)




from PIL import Image
import cv2
import pytesseract
import streamlit as st
import numpy as np
import io

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = 'model/Tesseract-OCR/tesseract.exe'

# Streamlit UI Enhancements
st.set_page_config(page_title="Vehicle Number Plate Detection", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚗 Vehicle Number Plate Detection 🏁</h1>", unsafe_allow_html=True)
st.write("---")  # Separator

def detect_image(uploaded_image):
    """Function to detect number plates from an image."""
    
    # Convert uploaded PIL image to NumPy array
    image = np.array(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    # Load the Haarcascade model
    vehicle_cascade = cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')

    # Detect number plates
    plates = vehicle_cascade.detectMultiScale(image, 1.1, 6)

    # Display detection count
    st.success(f"✅ Number of Number Plates Detected: {len(plates)}")
    
    # Draw rectangle and extract plates
    c = 1
    detected_plates = []
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red rectangle
        plate_area = image[y+7:y+h-7, x+7:x+w-7]

        # Perform OCR
        plate_text = pytesseract.image_to_string(plate_area, config='--psm 7').strip()
        
        # Display extracted plate
        st.image(plate_area, caption=f"🆔 Number Plate {c}", use_column_width=True)
        st.write(f"📌 **Detected Plate {c}:** `{plate_text}`")
        detected_plates.append(plate_text)
        c += 1

    return image, detected_plates

# UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_image = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)

    # Show progress while processing
    with st.spinner("🔍 Detecting Number Plates... Please wait..."):
        detected_image, plates = detect_image(image)

    # Show the processed image
    st.image(detected_image, caption="🖼️ Image with Detected Number Plates", use_column_width=True)

    # Provide download option
    img_bytes = io.BytesIO()
    Image.fromarray(detected_image).save(img_bytes, format="PNG")
    
    st.download_button("📥 Download Processed Image", img_bytes.getvalue(), "detected_number_plate.png", "image/png")

st.write("---")
st.markdown("<h5 style='text-align: center;'>👨‍💻 Developed with ❤️ using OpenCV, Streamlit, and Tesseract OCR</h5>", unsafe_allow_html=True)
