import argparse
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
import onnx
from onnx import helper, TensorProto
from PIL import Image


def export(model_builder, Model_Weights):
    base_dir = Path(f"models/keypoint_detection/{model_builder.__name__}")
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
        output_names=["boxes", "labels", "scores",
                      "keypoints", "keypoints_scores"],
        dynamic_axes={
            "boxes": {0: "detections"},
            "labels": {0: "detections"},
            "scores": {0: "detections"},
            "keypoints": {0: "detections"},
            "keypoints_scores": {0: "detections"}
        },
        export_params=True,
    )

    output_names = ["boxes", "labels", "scores",
                    "keypoints", "keypoints_scores"]

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
        match output_name:
            case "boxes":
                shape = [1, None, 4]
            case "keypoints":
                shape = [1, None, 17, 3]
            case "keypoints_scores":
                shape = [1, None, 17]
            case _:
                shape = [1, None]

        new_output = helper.make_tensor_value_info(
            name=output_name+"_unsqueezed",
            elem_type=TensorProto.INT64 if output_name == "labels" else TensorProto.FLOAT,
            shape=shape
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
    case "keypointrcnn_resnet50_fpn":
        from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
        export(keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
    case _:
        print("Model not found")
