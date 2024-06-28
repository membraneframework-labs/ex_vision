defmodule ExVision.KeypointDetection.KeypointRCNN_ResNet50_FPN do
  @moduledoc """
  Keypoint R-CNN model with a ResNet-50-FPN backbone, exported from torchvision.
  """
  use ExVision.Model.Definition.Ortex,
    model: "keypointrcnn_resnet50_fpn_keypoint_detector.onnx",
    categories: "priv/categories/no_person_or_person.json"

  require Logger

  alias ExVision.Types.BBoxWithKeypoints

  @typep output_t() :: [BBoxWithKeypoints.t()]

  @keypoints_names [
    :nose,
    :left_eye,
    :right_eye,
    :left_ear,
    :right_ear,
    :left_shoulder,
    :right_shoulder,
    :left_elbow,
    :right_elbow,
    :left_wrist,
    :right_wrist,
    :left_hip,
    :right_hip,
    :left_knee,
    :right_knee,
    :left_ankle,
    :right_ankle
  ]

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

  @impl true
  def preprocessing(img, _metadata) do
    ExVision.Utils.resize(img, {224, 224})
  end

  @impl true
  def postprocessing(
        %{
          "boxes_unsqueezed" => bboxes,
          "scores_unsqueezed" => scores,
          "labels_unsqueezed" => labels,
          "keypoints_unsqueezed" => keypoints_list,
          "keypoints_scores_unsqueezed" => keypoints_scores_list
        },
        metadata
      ) do
    categories = categories()

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

    keypoints_list =
      keypoints_list
      |> Nx.squeeze(axes: [0])
      |> Nx.multiply(Nx.tensor([scale_x, scale_y, 1]))
      |> Nx.round()
      |> Nx.as_type(:s64)
      |> Nx.to_list()

    keypoints_scores_list = keypoints_scores_list |> Nx.squeeze(axes: [0]) |> Nx.to_list()

    [bboxes, scores, labels, keypoints_list, keypoints_scores_list]
    |> Enum.zip()
    |> Enum.filter(fn {_bbox, score, _label, _keypoints, _keypoints_scores} -> score > 0.1 end)
    |> Enum.map(fn {[x1, y1, x2, y2], score, label, keypoints, keypoints_scores} ->
      keypoints =
        [keypoints, keypoints_scores]
        |> Enum.zip()
        |> Enum.map(fn {[x, y, _w], keypoint_score} -> %{x: x, y: y, score: keypoint_score} end)

      %BBoxWithKeypoints{
        x1: x1,
        x2: x2,
        y1: y1,
        y2: y2,
        score: score,
        label: Enum.at(categories, label),
        keypoints: [@keypoints_names, keypoints] |> Enum.zip() |> Map.new()
      }
    end)
  end
end
