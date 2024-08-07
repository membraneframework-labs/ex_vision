defmodule ExVision.Classification.EfficientNet_V2_M do
  @moduledoc """
  An object classifier based on EfficientNet_V2_M.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "efficientnet_v2_m_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  use ExVision.Classification.GenericClassifier

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({480, 480})
    |> NxImage.normalize(
      Nx.f32([0.485, 0.456, 0.406]),
      Nx.f32([0.229, 0.224, 0.225]),
      channels: :first
    )
  end
end
