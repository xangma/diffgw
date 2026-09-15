# External waveform data

This workflow keeps downloaded waveform datasets outside TorchWave's source
tree, wheels, and source distributions. TorchWave's code license remains MIT;
external datasets retain their own upstream terms.

The optional SEOBNRv4ROM v3 dataset is available from
[Zenodo record 14999310](https://zenodo.org/records/14999310). The data keeps
its upstream terms; this project does not make a legal clearance claim.
Install the optional `rom-official` dependency (`h5py`) when using it. The
external-data CLI reports and verifies files, and fetches only when explicitly
requested. Its search order is an explicit path, `TORCHWAVE_DATA_PATH`,
`LAL_DATA_PATH`, and the user cache (`$XDG_CACHE_HOME/torchwave/waveform-data`,
or `~/.cache/torchwave/waveform-data` when `XDG_CACHE_HOME` is unset):

```text
python -m torchwave.external_data list
python -m torchwave.external_data info seobnrv4-rom-v3
python -m torchwave.external_data verify seobnrv4-rom-v3
python -m torchwave.external_data fetch seobnrv4-rom-v3
```

The default fetch location is discovered automatically during waveform loading.
For a custom external directory, download there and configure lookup:

```sh
python -m torchwave.external_data fetch seobnrv4-rom-v3 --directory /path/to/cache
export TORCHWAVE_DATA_PATH=/path/to/cache
python -m torchwave.external_data verify seobnrv4-rom-v3
```

Alternatively, pass the downloaded file or its directory explicitly to
`OfficialSEOBNRv4ROM(model_path=...)`. Production loading checks the pinned SHA-256
before reading the ROM tables. A missing or corrupt file produces an error.

Fetch requires the explicit `fetch` command and a writable external directory;
no dataset is downloaded during installation or normal waveform evaluation.
No complete standalone XHM 122022 supplementary dataset is registered, so this
workflow cannot yet supply its missing calibration data. Download support does
not change the numerical parity results or replace missing waveform drivers.
