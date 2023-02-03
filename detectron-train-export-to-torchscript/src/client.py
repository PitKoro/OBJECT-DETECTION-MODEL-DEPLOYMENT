import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np


TRITON_URL="triton:8000"
image_file = '/app/dataset/test_imgs/balloons.jpg'
model_name = 'faster_rcnn'

def test_infer(req_id, image_file, model_name, print_output=False):
    img = np.array(Image.open(image_file))
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    # Define model's inputs
    inputs = []
    inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
    inputs[0].set_data_from_numpy(img)
    # Define model's outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
    outputs.append(httpclient.InferRequestedOutput('classes__1'))
    outputs.append(httpclient.InferRequestedOutput('scores__2'))
    outputs.append(httpclient.InferRequestedOutput('shape__3'))
    # Send request to Triton server
    triton_client = httpclient.InferenceServerClient(
        url=TRITON_URL, verbose=False)
    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)
    return outputs


result = test_infer(0, image_file, model_name)
print(result)