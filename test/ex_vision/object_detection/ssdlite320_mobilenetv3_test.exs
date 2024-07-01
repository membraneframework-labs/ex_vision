defmodule ExVision.ObjectDetection.Ssdlite320_MobileNetv3Test do
  use ExVision.Model.Case, module: ExVision.ObjectDetection.Ssdlite320_MobileNetv3
  use ExVision.TestUtils

  alias ExVision.Types.BBox

  @impl true
  def test_inference_result(result) do
    assert [%BBox{x1: 132, y1: 12, label: :cat, score: score}] = result
    assert_floats_equal(score, 1.0)
  end
end
