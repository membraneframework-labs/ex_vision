defmodule ExVision.InstanceSegmentation.MaskRCNN_ResNet50_FPN_V2_Test do
  use ExVision.Model.Case, module: ExVision.InstanceSegmentation.MaskRCNN_ResNet50_FPN_V2
  use ExVision.TestUtils
  alias ExVision.Types.BBoxWithMask

  @impl true
  def test_inference_result(result) do
    assert [%BBoxWithMask{x1: 129, y1: 15, label: :cat, score: score, mask: mask}] = result
    assert_floats_equal(score, 1.0)

    assert_floats_equal(nx_mean(mask), 0.37)
  end

  defp nx_mean(t), do: t |> Nx.mean() |> Nx.to_number()
end
