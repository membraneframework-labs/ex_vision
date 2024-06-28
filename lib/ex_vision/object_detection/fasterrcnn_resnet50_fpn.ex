defmodule ExVision.ObjectDetection.FasterRCNN_ResNet50_FPN do
  @moduledoc """
  FasterRCNN object detector with ResNet50 backbone and FPN detection head, exported from torchvision.
  """
  use ExVision.Model.Definition.Ortex,
    model: "fasterrcnn_resnet50_fpn_object_detector.onnx",
    categories: "priv/categories/coco_categories.json"

  use ExVision.ObjectDetection.GenericDetector

  require Logger

  @impl true
  def load(options \\ []) do
    if Keyword.has_key?(options, :batch_size) do
      Logger.warning(
        "`:max_batch_size` was given, but this model can only process batch of size 1. Overriding"
      )
    end

    options
    |> Keyword.put(:batch_size, 1)
    |> default_model_load()
  end
end
