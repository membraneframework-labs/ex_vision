defmodule ExVision.Model.Case do
  @moduledoc false
  @img_path "test/assets/cat.jpg"

  @callback test_inference_result(result :: any()) :: any()

  defmacro __using__(opts) do
    opts = Keyword.validate!(opts, [:module])

    quote do
      use ExUnit.Case, async: true
      # use ExVision.TestUtils.MockCacheServer
      @behaviour ExVision.Model.Case

      setup_all do
        {:ok, model} = unquote(opts[:module]).load()
        [model: model]
      end

      test "load/0", %{model: model} do
        assert model
      end

      test "inference", %{model: model} do
        model
        |> unquote(opts[:module]).run(unquote(@img_path))
        |> test_inference_result()
      end

      test "inference for batch", %{model: model} do
        model
        |> unquote(opts[:module]).run([unquote(@img_path), unquote(@img_path)])
        |> Enum.each(&test_inference_result/1)
      end

      test "child_spec/1" do
        assert spec = unquote(opts[:module]).child_spec()
      end

      describe "stateful/process workflow" do
        setup ctx do
          name = String.to_atom("#{__MODULE__}#{ctx[:test]}")
          model = ctx[:model]

          {:ok, _supervisor} =
            Supervisor.start_link(
              [unquote(opts[:module]).child_spec(name: name)],
              strategy: :one_for_one
            )

          [name: name]
        end

        test "inference", %{name: name} do
          name
          |> unquote(opts[:module]).batched_run(unquote(@img_path))
          |> test_inference_result()
        end

        test "inference for batch", %{name: name} do
          name
          |> unquote(opts[:module]).batched_run([unquote(@img_path), unquote(@img_path)])
          |> Enum.each(&test_inference_result/1)
        end
      end

      test "stateful/process workflow accepts options" do
        options = [
          name: __MODULE__.TestProcess1,
          batch_size: 8,
          batch_timeout: 10,
          partitions: true
        ]

        child_spec = {unquote(opts[:module]), options}

        assert {:ok, _supervisor} =
                 Supervisor.start_link([child_spec], strategy: :one_for_one, restarts: :none)

        assert unquote(opts[:module]).batched_run(
                 __MODULE__.TestProcess1,
                 unquote(@img_path)
               )
      end
    end
  end
end
