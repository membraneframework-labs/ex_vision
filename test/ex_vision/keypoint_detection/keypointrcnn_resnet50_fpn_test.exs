defmodule ExVision.KeypointDetection.KeypointRCNN_ResNet50_FPNTest do
  use ExVision.Model.Case, module: ExVision.KeypointDetection.KeypointRCNN_ResNet50_FPN
  use ExVision.TestUtils
  alias ExVision.Types.BBoxWithKeypoints

  @impl true
  def test_inference_result(result) do
    assert [
             %BBoxWithKeypoints{
               x1: 113,
               y1: 15,
               label: :person,
               score: score1,
               keypoints: keypoints
             },
             %BBoxWithKeypoints{
               x1: 141,
               y1: 167,
               label: :person,
               score: score2
             }
           ] = result

    assert_floats_equal(score1, 0.46)
    assert_floats_equal(score2, 0.29)

    assert max_keypoint_score(keypoints) < 5
  end

  defp max_keypoint_score(keypoints) do
    keypoints |> Enum.map(fn {_name, %{score: score}} -> score end) |> Enum.max()
  end
end
