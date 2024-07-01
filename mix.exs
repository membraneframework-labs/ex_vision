defmodule ExVision.Mixfile do
  use Mix.Project

  @version "0.2.0"
  @github_url "https://github.com/membraneframework-labs/ex_vision/"

  def project do
    [
      app: :ex_vision,
      version: @version,
      elixir: "~> 1.16",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: dialyzer(),

      # hex
      description: "A collection of ONNX vision AI models with wrappers based on Ortex",
      package: package(),

      # docs
      name: "Ex Vision",
      source_url: @github_url,
      docs: docs(),
      homepage_url: "https://hexdocs.pm/ex_vision"
    ]
  end

  def application do
    [
      included_applications: [:ex_vision],
      mod: {ExVision, []},
      extra_applications: []
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  defp deps do
    [
      # TODO: change the `>= 0.0.0` dependencies to concrete versions
      {:nx, ">= 0.0.0"},
      {:ortex, ">= 0.0.0"},
      {:nx_image, "~> 0.1.2"},
      {:bunch, "~> 1.6", runtime: false},
      {:axon, "~> 0.6.1"},
      {:exla, ">= 0.0.0"},
      {:image, ">= 0.0.0"},
      {:req, ">= 0.0.0"},
      {:mimic, "~> 1.7", only: :test},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:dialyxir, ">= 0.0.0", only: :dev, runtime: false},
      {:credo, ">= 0.0.0", only: [:dev, :test], runtime: false}
    ]
  end

  defp dialyzer() do
    opts = [
      flags: [:error_handling]
    ]

    if System.get_env("CI") == "true" do
      # Store PLTs in cacheable directory for CI
      [plt_local_path: "priv/plts", plt_core_path: "priv/plts"] ++ opts
    else
      opts
    end
  end

  defp package do
    [
      maintainers: ["Software Mansion"],
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @github_url,
        "Software Mansion" => "https://www.swmansion.com"
      }
    ]
  end

  @tutorials Path.wildcard("examples/*.livemd")
  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "LICENSE"
        | @tutorials
      ],
      groups_for_extras: [
        Tutorials: @tutorials
      ],
      groups_for_modules: [
        Models: [
          ExVision.Classification.MobileNetV3Small,
          ExVision.Classification.EfficientNet_V2_S,
          ExVision.Classification.EfficientNet_V2_M,
          ExVision.Classification.EfficientNet_V2_L,
          ExVision.Classification.SqueezeNet1_1,
          ExVision.SemanticSegmentation.DeepLabV3_MobileNetV3,
          ExVision.ObjectDetection.Ssdlite320_MobileNetv3,
          ExVision.ObjectDetection.FasterRCNN_ResNet50_FPN
        ],
        Types: [
          ExVision.Types,
          ExVision.Types.BBox,
          ExVision.Types.ImageMetadata
        ],
        "Protocols and Behaviours": [
          ExVision.Model,
          ExVision.Model.Definition,
          ExVision.Model.Definition.Ortex
        ]
      ],
      nest_modules_by_prefix: [
        ExVision.Model,
        ExVision.Model.Definition,
        ExVision.Types,
        ExVision.Classification,
        ExVision.SemanticSegmentation,
        ExVision.ObjectDetection
      ],
      formatters: ["html"],
      source_ref: "v#{@version}"
    ]
  end
end
