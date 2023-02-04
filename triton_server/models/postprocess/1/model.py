import triton_python_backend_utils as pb_utils


class TritonPythonModel:
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
            out_tensors = []
            for name in ['bboxes', 'classes', 'scores']:
                tensor = pb_utils.Tensor('post_' + name, predictions[name])
                out_tensors.append(tensor)
            response = pb_utils.InferenceResponse(output_tensors=out_tensors)
            responses.append(response)
        return responses