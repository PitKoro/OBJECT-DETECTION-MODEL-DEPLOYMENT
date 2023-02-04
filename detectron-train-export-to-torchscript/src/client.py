import argparse
import time
import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np


def test_infer(image_file, model_name='faster_rcnn'):
    img = np.array(Image.open(image_file))
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    inputs = []
    inputs.append(httpclient.InferInput('image__0', img.shape, "UINT8"))
    inputs[0].set_data_from_numpy(img)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('bboxes__0'))
    outputs.append(httpclient.InferRequestedOutput('classes__1'))
    outputs.append(httpclient.InferRequestedOutput('scores__2'))
    outputs.append(httpclient.InferRequestedOutput('shape__3'))

    triton_client = httpclient.InferenceServerClient(
        url="triton:8000", verbose=False)
    results = triton_client.infer(model_name, inputs=inputs, outputs=outputs)
    response_info = results.get_response()
    outputs = {}
    for output_info in response_info['outputs']:
        output_name = output_info['name']
        outputs[output_name] = results.as_numpy(output_name)
        
    return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--num-reqs', default='1')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_file = args.image
    n_reqs = int(args.num_reqs)


    times = []
    for i in tqdm(range(n_reqs)):
        s = time.time()
        test_infer(image_file)
        e = time.time()
        times.append(e - s)

    print('Среднее время обработки запроса:', sum(times)/len(times), 's')
   