echo "Downloading weights..."
wget https://github.com/PitKoro/SberCloudTestTask/releases/download/triton/faster_rcnn_torchscript_model.tar.xz
wget https://github.com/PitKoro/SberCloudTestTask/releases/download/triton/faster_rcnn_R_50.tar.xz
echo "Extracting files..."
tar -xf faster_rcnn_torchscript_model.tar.xz -C './triton_server/models/faster_rcnn/1'
mkdir -p ./detectron-train-export-to-torchscript/checkpoints/detectron2
tar -xf faster_rcnn_R_50.tar.xz -C './detectron-train-export-to-torchscript/checkpoints/detectron2'
echo "Deleting archives..."
rm faster_rcnn_torchscript_model.tar.xz
rm faster_rcnn_R_50.tar.xz
echo "DONE"