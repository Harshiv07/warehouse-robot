from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = Image.open("image1.png")
text = pytesseract.image_to_string(img)
print(f"Recognized Text from OCR: {text}")
