import numpy as np
import onnxruntime


class AVESOnnxModel:
    """Wrapper class for ONNX models."""

    def __init__(self, model_path: str):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = {self.ort_session.get_inputs()[0].name: inputs}
        outputs = self.ort_session.run(None, inputs)
        return outputs[0]
