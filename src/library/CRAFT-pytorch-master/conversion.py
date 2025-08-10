import torch as Torch
from craft import CRAFT
from test import copyStateDict
import onnx as Onnx
from onnx_tf.backend import prepare

# From torch
torchModel = CRAFT(pretrained=False)
modelStateDict = copyStateDict(
    Torch.load(
        "/home/app/src/library/CRAFT-pytorch-master/model/craft_mlt_25k.pth",
        map_location="cpu",
    )
)
torchModel.load_state_dict(modelStateDict)

# To onnx
input = Torch.randn(1, 3, 480, 640)
# i_names = ["input_1"]
Torch.onnx.export(
    torchModel,
    input,
    "/home/app/src/library/CRAFT-pytorch-master/model/craft_mlt_25k.onnx",
)

# To tensorflow
onnx_model = Onnx.load(
    "/home/app/src/library/CRAFT-pytorch-master/model/craft_mlt_25k.onnx"
)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("output_path")
