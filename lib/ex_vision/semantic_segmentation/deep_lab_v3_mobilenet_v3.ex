defmodule ExVision.SemanticSegmentation.DeepLabV3_MobileNetV3 do
  @moduledoc """
  A semantic segmentation model for MobileNetV3 Backbone. Exported from torchvision.
  """
  use ExVision.Model.Definition.Ortex,
    model: "deeplab_v3_mobilenetv3_segmentation.onnx",
    categories: "priv/categories/coco_with_voc_labels_categories.json"

  @type output_t() :: %{category_t() => Nx.Tensor.t()}

  @impl true
  def preprocessing(img, _metdata) do
    ExVision.Utils.resize(img, {224, 224})
  end

  @impl true
  def postprocessing(%{"output" => out}, metadata) do
    cls_per_pixel =
      out
      |> Nx.backend_transfer()
      |> NxImage.resize(metadata.original_size, channels: :first)
      |> Nx.squeeze()
      |> Axon.Activations.softmax(axis: [0])
      |> Nx.argmax(axis: 0)

    categories()
    |> Enum.with_index()
    |> Map.new(fn {category, i} ->
      {category, cls_per_pixel |> Nx.equal(i)}
    end)
  end
end
