defmodule ExVision.Cache do
  @moduledoc false
  alias ExVision.Cache.PubSub

  # Module responsible for handling model file caching

  require Logger

  alias ExVision.Cache.PubSub

  @default_cache_path Application.compile_env(:ex_vision, :cache_path, "/tmp/ex_vision/cache")
  defp get_cache_path() do
    Application.get_env(:ex_vision, :cache_path, @default_cache_path)
  end

  @default_server_url Application.compile_env(
                        :ex_vision,
                        :server_url,
                        URI.new!("https://ai.swmansion.com/exvision/files")
                      )
  defp get_server_url() do
    Application.get_env(:ex_vision, :server_url, @default_server_url)
  end

  @type lazy_get_option_t() ::
          {:cache_path, Path.t()} | {:server_url, String.t() | URI.t()} | {:force, boolean()}

  @spec child_spec(keyword())
  def child_spec(_opts) do
    Registry.child_spec(name: __MODULE__)
  end

  @doc """
  Lazily evaluate the path from the cache directory.
  It will only download the file if it's missing or the `force: true` option is given.
  """
  @spec lazy_get(Path.t(), options :: [lazy_get_option_t()]) ::
          {:ok, Path.t()} | {:error, reason :: atom()}
  def lazy_get(path, options \\ []) do
    options =
      Keyword.validate!(options,
        cache_path: get_cache_path(),
        server_url: get_server_url(),
        force: false
      )

    cache_path = Path.join(options[:cache_path], path)
    ok? = File.exists?(cache_path)

    if ok? and not options[:force] do
      Logger.debug("Found existing cache entry for #{path}. Loading.")
      {:ok, cache_path}
    else
      with {:ok, server_url} <- URI.new(options[:server_url]) do
        download_url = URI.append_path(server_url, ensure_backslash(path))
        download_file(download_url, cache_path)
      end
    end
  end

  defp create_download_job(url, cache_path) do
    key = {url, cache_path}

    spawn(fn ->
      PubSub.notify(__MODULE__, key, download_file(url, cache_path))
    end)

    PubSub.subscribe(__MODULE__, key)
  end

  @spec download_file(URI.t(), Path.t()) ::
          {:ok, Path.t()} | {:error, reason :: any()}
  defp download_file(url, cache_path) do
    with :ok <- cache_path |> Path.dirname() |> File.mkdir_p(),
         tmp_file_path = cache_path <> ".unconfirmed",
         tmp_file = File.stream!(tmp_file_path),
         :ok <- do_download_file(url, tmp_file),
         :ok <- validate_download(tmp_file_path),
         :ok <- File.rename(tmp_file_path, cache_path) do
      {:ok, cache_path}
    end
  end

  defp validate_download(path) do
    if File.exists?(path),
      do: :ok,
      else: {:error, :download_failed}
  end

  @spec do_download_file(URI.t(), File.Stream.t()) :: :ok | {:error, reason :: any()}
  defp do_download_file(%URI{} = url, %File.Stream{path: target_file_path} = target_file) do
    Logger.debug("Downloading file from `#{url}` and saving to `#{target_file_path}`")

    case make_get_request(url, raw: true, into: target_file) do
      {:ok, _resp} ->
        :ok

      {:error, reason} = error ->
        Logger.error("Failed to download the file due to #{inspect(reason)}")
        File.rm(target_file_path)
        error
    end
  end

  defp make_get_request(url, options) do
    url
    |> Req.get(options)
    |> case do
      {:ok, %Req.Response{status: 200}} = resp ->
        resp

      {:ok, %Req.Response{status: 404}} ->
        {:error, :doesnt_exist}

      {:ok, %Req.Response{status: status}} ->
        Logger.warning("Request has failed with status #{status}")
        {:error, :server_error}

      {:error, %Mint.TransportError{reason: reason}} ->
        {:error, reason}

      {:error, _error} ->
        {:error, :connection_failed}
    end
  end

  defp ensure_backslash("/" <> _rest = path), do: path
  defp ensure_backslash(path), do: "/" <> path
end

defmodule ExVision.Cache.PubSub do
  @moduledoc false

  @spec subscribe(term(), Path.t()) :: :ok
  def subscribe(registry, key) do
    Registry.register(registry, key, [])

    receive do
      {^registry, :notification, result} ->
        Registry.unregister(registry, key)
        result
    end
  end

  @spec notify(term(), Path.t()) :: :ok
  def notify(registry, key, result) do
    Registry.dispatch(registry, key, fn entries ->
      for {pid, _value} <- entries, do: send(pid, {registry, :notification, result})
    end)
  end
end
