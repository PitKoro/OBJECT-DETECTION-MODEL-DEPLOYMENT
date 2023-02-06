echo "Downloading weights..."
wget https://github.com/PitKoro/SberCloudTestTask/releases/download/triton/faster_rcnn_torchscript_model.tar.xz
wget https://github.com/PitKoro/SberCloudTestTask/releases/download/triton/faster_rcnn_R_50.tar.xz
echo "Extracting files..."
tar -xf faster_rcnn_torchscript_model.tar.xz -C './triton/models/faster_rcnn/1'
mkdir -p ./jupyter/checkpoints/detectron2
tar -xf faster_rcnn_R_50.tar.xz -C './jupyter/checkpoints/detectron2'
mkdir -p ./kserve/weights
tar -xf faster_rcnn_R_50.tar.xz -C './kserve/weights'
echo "Deleting archives..."
rm faster_rcnn_torchscript_model.tar.xz
rm faster_rcnn_R_50.tar.xz
echo "DONE"