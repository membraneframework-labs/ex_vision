import argparse
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
import onnx
from onnx import helper, TensorProto
from PIL import Image


def export(model_builder, Model_Weights, output_names):
    base_dir = Path(f"models/object_detection/{model_builder.__name__}")
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
        output_names=output_names,
        dynamic_axes={
            "boxes": {0: "detections"},
            "labels": {0: "detections"},
            "scores": {0: "detections"},
        },
        export_params=True,
    )

    model = onnx.load(str(model_file))

    nodes = []
    for output_name in output_names:
        axes_init = helper.make_tensor(
            name=output_name+"_axes",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
        model.graph.initializer.append(axes_init)

        node = helper.make_node(
            op_type="Unsqueeze",
            inputs=[output_name, output_name+"_axes"],
            outputs=[output_name+"_unsqueezed"]
        )
        nodes.append(node)

    model.graph.node.extend(nodes)

    new_outputs = []
    for output_name in output_names:
        new_output = helper.make_tensor_value_info(
            name=output_name+"_unsqueezed",
            elem_type=TensorProto.INT64 if output_name == "labels" else TensorProto.FLOAT,
            shape=[1, None, 4] if output_name == "boxes" else [1, None]
        )
        new_outputs.append(new_output)

    model.graph.output.extend(new_outputs)

    for output_name in output_names:
        old_output = next(
            i for i in model.graph.output if i.name == output_name)
        model.graph.output.remove(old_output)

    onnx.save(model, str(model_file))


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

match(args.model):
    case "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        export(
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
            ["boxes", "labels", "scores"]
        )
    case "ssdlite320_mobilenet_v3_large":
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        export(
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
            ["boxes", "scores", "labels"]
        )
    case _:
        print("Model not found")
