import cv2
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

def test_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    try:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame from webcam")
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = mp_face_detection.process(rgb_frame)
        
        if not results.detections:
            logger.warning("No faces detected")
        else:
            logger.info(f"Detected {len(results.detections)} faces")
            
            # Draw detection on frame
            height, width = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the frame
        cv2.imwrite('test_frame.jpg', frame)
        logger.info("Saved test frame to test_frame.jpg")
        
    except Exception as e:
        logger.error(f"Error during webcam test: {e}")
    
    finally:
        cap.release()

if __name__ == '__main__':
    test_webcam()