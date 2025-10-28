import numpy as np
import cv2
from pathlib import Path
import tflite_runtime.interpreter as tflite

class MobileFaceNetEmbedder:
    def __init__(self, model_path: str, use_tta: bool = False):
        model_path = str(Path(model_path).expanduser().resolve())
        self.interp = tflite.Interpreter(model_path=model_path, num_threads=2)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]['index']
        self.out_idx = self.interp.get_output_details()[0]['index']
        self.input_shape = self.interp.get_input_details()[0]['shape']  # [1,112,112,3] kỳ vọng
        self.use_tta = use_tta
        
        # Warmup: chạy một lần để tối ưu cache
        dummy = np.zeros((1, 112, 112, 3), dtype=np.float32)
        self.interp.set_tensor(self.in_idx, dummy)
        self.interp.invoke()

    @staticmethod
    def _prewhiten(x: np.ndarray) -> np.ndarray:
        # Chuẩn hoá nhẹ theo kiểu FaceNet/ArcFace
        mean = x.mean()
        std = x.std()
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = (x - mean) / std_adj
        return y

    def preprocess(self, bgr_face: np.ndarray) -> np.ndarray:
        # Align đơn giản: resize về 112x112, BGR->RGB, scale [-1,1] hoặc prewhiten
        img = cv2.resize(bgr_face, (112,112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Cách 1: prewhiten
        rgb = self._prewhiten(rgb)
        # expand batch
        rgb = np.expand_dims(rgb, axis=0)
        return rgb

    def _get_embedding(self, preprocessed: np.ndarray) -> np.ndarray:
        """Get embedding from preprocessed image"""
        self.interp.set_tensor(self.in_idx, preprocessed)
        self.interp.invoke()
        emb = self.interp.get_tensor(self.out_idx).astype(np.float32).reshape(-1)
        # L2 normalize
        n = np.linalg.norm(emb) + 1e-9
        emb = emb / n
        return emb

    def __call__(self, bgr_face: np.ndarray) -> np.ndarray:
        # Original embedding
        inp = self.preprocess(bgr_face)
        emb_orig = self._get_embedding(inp)
        
        if not self.use_tta:
            return emb_orig
        
        # TTA: flip horizontal
        try:
            bgr_flipped = cv2.flip(bgr_face, 1)  # Horizontal flip
            inp_flipped = self.preprocess(bgr_flipped)
            emb_flipped = self._get_embedding(inp_flipped)
            
            # Average của 2 embeddings
            emb_avg = (emb_orig + emb_flipped) / 2.0
            
            # Re-normalize
            n = np.linalg.norm(emb_avg) + 1e-9
            emb_avg = emb_avg / n
            
            return emb_avg
        except Exception:
            # Fallback nếu TTA fail
            return emb_orig
