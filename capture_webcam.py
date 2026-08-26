import cv2
import time

def main():
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Allow camera to warm up
    time.sleep(1)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("webcam_image.jpg", frame)
        print("Successfully captured webcam_image.jpg")
    else:
        print("Failed to capture from webcam.")
    cap.release()

if __name__ == "__main__":
    main()
