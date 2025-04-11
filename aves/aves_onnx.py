import numpy as np
import onnxruntime


class AVESOnnxModel:
    """Wrapper class for ONNX models.

    Arguments
    ---------
    model_path: str
        Path to the ONNX model file.

    Examples
    --------
    >>> model = AVESOnnxModel("model.onnx")
    >>> inputs = np.random.rand(1, 16000)
    >>> outputs = model(inputs)
    """

    def __init__(self, model_path: str):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = {self.ort_session.get_inputs()[0].name: inputs}
        outputs = self.ort_session.run(None, inputs)
        return outputs[0]
