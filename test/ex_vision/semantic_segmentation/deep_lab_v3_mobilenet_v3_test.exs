defmodule ExVision.SemanticSegmentation.DeepLabV3_MobileNetV3Test do
  use ExVision.Model.Case, module: ExVision.SemanticSegmentation.DeepLabV3_MobileNetV3
  use ExVision.TestUtils

  @impl true
  def test_inference_result(result) do
    assert %{cat: cat, __background__: background} = result,
           "The result doesn't contain required classes"

    assert_floats_equal(nx_mean(cat) + nx_mean(background), 1.0)
    assert_floats_equal(nx_mean(cat), 0.36)
  end

  defp nx_mean(t), do: t |> Nx.mean() |> Nx.to_number()
end
