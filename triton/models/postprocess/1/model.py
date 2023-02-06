import os
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    
    def __init__(self):
        super().__init__()
        self.labels = self.get_labels_from_file()


    def get_labels_from_file(self, file_path=f"{os.path.dirname(os.path.realpath(__file__))}/labels.txt"):
        with open(file_path, "r") as labels_file:
            labels = labels_file.read().split('\n')
        return labels


    def get_inputs(self, request):
        input_tensor_names = ['bboxes', 'classes', 'scores', 'shape']
        inputs = {
            tensor_name: pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()
                  for tensor_name in input_tensor_names
        }
        return inputs


    def execute(self, requests):
        responses = []
        for request in requests:
            predictions = self.get_inputs(request)
            predicted_labels = []
            for i in predictions['classes']:
                predicted_labels.append(self.labels[i])
            predictions['classes'] = np.array(predicted_labels, dtype=object)
            out_tensors = []
            for name in ['bboxes', 'classes', 'scores']:
                tensor = pb_utils.Tensor('post_' + name, predictions[name])
                out_tensors.append(tensor)
            response = pb_utils.InferenceResponse(output_tensors=out_tensors)
            responses.append(response)
        return responses