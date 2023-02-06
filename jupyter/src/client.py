import argparse
import time
import tritonclient.http as httpclient
from tqdm import tqdm
from PIL import Image
import numpy as np


def test_infer(image_file, model_name='infer_pipeline'):
    with open(image_file, 'rb') as fi:
        image_bytes = fi.read()
    image_bytes = np.array([image_bytes], dtype=np.bytes_)

    inputs = []
    inputs.append(httpclient.InferInput('IMAGE_BYTES', image_bytes.shape, "BYTES"))
    inputs[0].set_data_from_numpy(image_bytes)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput('BBOXES'))
    outputs.append(httpclient.InferRequestedOutput('CLASSES'))
    outputs.append(httpclient.InferRequestedOutput('SCORES'))

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
   