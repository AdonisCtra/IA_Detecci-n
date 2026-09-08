try:
    import onnxruntime as ort
    _HAS_ONNXRT = True
except Exception:
    ort = None
    _HAS_ONNXRT = False

import numpy as np
import cv2


class VerifierONNX:
    def __init__(self, onnx_path: str, provider: str = None):
        if not _HAS_ONNXRT:
            raise RuntimeError('onnxruntime no está instalado')
        providers = None
        if provider:
            providers = [provider]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, image):
        # expects RGB numpy array
        im = cv2.resize(image, (224, 224))
        im = im.astype('float32') / 255.0
        im = np.transpose(im, (2, 0, 1))
        im = np.expand_dims(im, 0)
        return im

    def predict_probs(self, crops: list, batch_size: int = 8):
        results = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            arrs = [self.preprocess(b) for b in batch]
            x = np.vstack(arrs)
            out = self.sess.run([self.output_name], {self.input_name: x})[0]
            # softmax for output to probabilities
            ex = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = ex[:, 1] / np.sum(ex, axis=1)
            results.extend(probs.tolist())
        return results
