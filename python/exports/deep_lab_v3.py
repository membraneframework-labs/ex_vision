from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
import torch
import json
from pathlib import Path

base_dir = Path("models/segmentation/deeplab_v3_mobilenetv3")
base_dir.mkdir(parents=True, exist_ok=True)

model_file = base_dir / "model.onnx"
categories_file = base_dir / "categories.json"

weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.eval()

categories = weights.meta["categories"]
transforms = weights.transforms()

with open(categories_file, "w") as f:
    json.dump(categories, f)

onnx_input = torch.rand(1, 3, 224, 224)


from torchvision.io.image import read_image

cat = read_image("examples/files/cat.jpg")
batch = transforms(cat).unsqueeze(0)
outputs = model(batch)

torch.onnx.export(
    model,
    batch,
    str(model_file),
    verbose=False,
    input_names=["input"],
    output_names=["output", "aux"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "width", 3: "height"},
        "output": {0: "batch_size", 2: "width", 3: "height"},
    },
    export_params=True,
)

print(transforms)
print(batch.shape)
print(outputs)
