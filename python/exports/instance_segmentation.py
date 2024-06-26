import argparse
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
import onnx
from onnx import helper, TensorProto
from PIL import Image


def export(model_builder, Model_Weights):
    base_dir = Path(f"models/instance_segmentation/{model_builder.__name__}")
    base_dir.mkdir(parents=True, exist_ok=True)

    model_file = base_dir / "model.onnx"
    categories_file = base_dir / "categories.json"

    weights = Model_Weights.DEFAULT
    model = model_builder(weights=weights)
    model.eval()

    categories = weights.meta["categories"]
    transforms = weights.transforms()

    with open(categories_file, "w") as f:
        json.dump(categories, f)

    onnx_input = to_tensor(Image.open("test/assets/cat.jpg")).unsqueeze(0)
    onnx_input = resize(onnx_input, [224, 224])
    onnx_input = transforms(onnx_input)

    torch.onnx.export(
        model,
        onnx_input,
        str(model_file),
        verbose=False,
        input_names=["input"],
        output_names=["boxes", "labels", "scores", "masks"],
        dynamic_axes={
            "boxes": {0: "detections"},
            "labels": {0: "detections"},
            "scores": {0: "detections"},
            "masks": {0: "detections"},
        },
        export_params=True,
    )

    model = onnx.load(str(model_file))

    prev_names = ["boxes", "labels", "scores", "masks"]

    nodes = []
    for data in prev_names:
        axes_init = helper.make_tensor(
            name=data+"_axes",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
        model.graph.initializer.append(axes_init)

        node = helper.make_node(
            op_type="Unsqueeze",
            inputs=[data, data+"_axes"],
            outputs=[data+"_unsqueezed"]
        )
        nodes.append(node)

    model.graph.node.extend(nodes)

    new_outputs = []
    for data in prev_names:
        match data:
            case "boxes":
                shape = [1, None, 4]
            case "masks":
                shape = [1, None, 1, 224, 224]
            case _:
                shape = [1, None]

        new_output = helper.make_tensor_value_info(
            name=data+"_unsqueezed",
            elem_type=TensorProto.INT64 if data == "labels" else TensorProto.FLOAT,
            shape=shape
        )
        new_outputs.append(new_output)

    model.graph.output.extend(new_outputs)

    for data in prev_names:
        old_output = next(i for i in model.graph.output if i.name == data)
        model.graph.output.remove(old_output)

    onnx.save(model, str(model_file))


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

match(args.model):
    case "maskrcnn_resnet50_fpn_v2":
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        export(maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights)
    case _:
        print("Model not found")
