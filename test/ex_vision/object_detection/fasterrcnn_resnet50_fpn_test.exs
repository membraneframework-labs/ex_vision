defmodule ExVision.ObjectDetection.FasterRCNN_ResNet50_FPN_Test do
  use ExVision.Model.Case, module: ExVision.ObjectDetection.FasterRCNN_ResNet50_FPN
  use ExVision.TestUtils
  alias ExVision.Types.BBox

  @impl true
  def test_inference_result(result) do
    assert [%BBox{x1: 135, y1: 22, label: :cat, score: score}] = result
    assert_floats_equal(score, 1.0)
  end
end
