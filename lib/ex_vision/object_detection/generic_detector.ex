defmodule ExVision.ObjectDetection.GenericDetector do
  @moduledoc false

  # Contains a default implementation of pre and post processing for TorchVision detectors
  # To use: `use ExVision.ObjectDetection.GenericDetector`

  require Logger

  alias ExVision.Types.{BBox, ImageMetadata}

  @typep output_t() :: [BBox.t()]

  @spec preprocessing(Nx.Tensor.t(), ImageMetadata.t()) :: Nx.Tensor.t()
  def preprocessing(img, _metadata) do
    ExVision.Utils.resize(img, {224, 224})
  end

  @spec postprocessing(map(), ImageMetadata.t(), [atom()]) :: output_t()
  def postprocessing(
        %{
          "boxes_unsqueezed" => bboxes,
          "scores_unsqueezed" => scores,
          "labels_unsqueezed" => labels
        },
        metadata,
        categories
      ) do
    {h, w} = metadata.original_size
    scale_x = w / 224
    scale_y = h / 224

    bboxes =
      bboxes
      |> Nx.squeeze(axes: [0])
      |> Nx.multiply(Nx.tensor([scale_x, scale_y, scale_x, scale_y]))
      |> Nx.round()
      |> Nx.as_type(:s64)
      |> Nx.to_list()

    scores = scores |> Nx.squeeze(axes: [0]) |> Nx.to_list()
    labels = labels |> Nx.squeeze(axes: [0]) |> Nx.to_list()

    [bboxes, scores, labels]
    |> Enum.zip()
    |> Enum.filter(fn {_bbox, score, _label} -> score > 0.1 end)
    |> Enum.map(fn {[x1, y1, x2, y2], score, label} ->
      %BBox{
        x1: x1,
        x2: x2,
        y1: y1,
        y2: y2,
        score: score,
        label: Enum.at(categories, label)
      }
    end)
  end

  defmacro __using__(_opts) do
    quote do
      @typedoc """
      A type describing output of `run/2` as a list of a bounding boxes.

      Each bounding box describes the location of the object indicated by the `label`.
      It also provides the `score` field marking the probability of the prediction.
      Bounding boxes with very low scores should most likely be ignored.
      """
      @type output_t() :: [BBox.t()]

      @impl true
      defdelegate preprocessing(image, metadata), to: ExVision.ObjectDetection.GenericDetector

      @impl true
      @spec postprocessing(map(), ExVision.Types.ImageMetadata.t()) :: output_t()
      def postprocessing(output, metadata) do
        ExVision.ObjectDetection.GenericDetector.postprocessing(output, metadata, categories())
      end

      defoverridable preprocessing: 2, postprocessing: 2
    end
  end
end
