from app.main import EmotionDetector
import numpy as np

if __name__ == '__main__':
    ed = EmotionDetector()
    print('Model input shape cached:', ed.model_input_shape)
    # create a dummy RGB face according to model_input_shape
    if ed.model_input_shape and len(ed.model_input_shape) == 4:
        s = ed.model_input_shape
        if ed._is_nchw():
            _, c, h, w = s
            h = int(h) if h else 224
            w = int(w) if w else 224
            face = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        else:
            _, h, w, c = s
            h = int(h) if h else 48
            w = int(w) if w else 48
            face = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        face = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)

    res = ed.detect_emotion(face)
    print('Detect result:', res)
