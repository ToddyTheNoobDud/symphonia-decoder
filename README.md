# Symphonia Decoder

A native Node.js addon for decoding audio with Symphonia. It also lets NodeLink extract audio packets from MP4 files without loading the whole container into memory.

## Install

```sh
npm install @toddynnn/symphonia-decoder
```

The npm package loads a prebuilt binding for your platform when one is available.

## Extract AAC from an MP4 or M4A file

Use `demuxMp4Aac()` when you already have the complete file in memory. It returns the AAC packets, their decode timestamps in seconds, and the track's AudioSpecificConfig.

```js
import { readFile } from 'node:fs/promises'
import { demuxMp4Aac } from '@toddynnn/symphonia-decoder'

const { packets, sampleRate, channels } = demuxMp4Aac(await readFile('track.m4a'))

console.log({ packets: packets.length, sampleRate, channels })
```

For streaming or seeking, use `openMp4Audio(size, read, seekSeconds?)`. The async `read(error, [offset, length])` callback supplies bytes from a file or HTTP Range request. It must return the requested number of bytes. Pass `null` for a forward-only MP4 whose index comes before the media. In that mode, the callback can return a short read at EOF.

- `readPackets(limit)` returns compressed AAC or Opus packets. `readPcm()` decodes ALAC to 48 kHz stereo PCM.
- `audioRanges` contains exact packet offsets for supported MP4 layouts. HTTP clients can use them to batch reads. An empty array means the reader uses ordinary byte requests.
- Cancel pending transport reads before calling `close()`.

## Decode and resample audio

- `SymphoniaDecoder` accepts compressed chunks through `push()`. Call `initialize()` once enough input is available, then call `decode()` to read PCM. At EOF, call `closeInput()` and drain the decoder before releasing it with `free()`.
- `PcmResampler(inputRate, quality)` takes interleaved floats and a channel count through `process()`. It returns 48 kHz stereo PCM as signed 16-bit little-endian bytes. Call `finish()` at EOF to drain queued samples, or `close()` to discard them.

## Build from source

Install Rust, CMake, and a C compiler, then run:

```sh
corepack yarn install
corepack yarn build
corepack yarn lint
```

The build produces a platform-specific `toddy-decoder.*.node` binding. On Windows, use a Visual Studio developer shell with Ninja on PATH. If MSVC fails to write a PDB, try a release build with a short ASCII target path:

```powershell
$env:CMAKE_GENERATOR = 'Ninja'
$target = Join-Path ([IO.Path]::GetTempPath()) 'symphonia-build'
node .\node_modules\@napi-rs\cli\dist\cli.js build --platform --release --target-dir $target
```

## Release

Set the package version before regenerating the bindings:

```sh
npm version <version> --no-git-tag-version --ignore-scripts
corepack yarn build
```

Commit the version and generated bindings with just the version number as the commit message, for example `1.3.1`. CI builds the platform packages and publishes them after the commit is pushed to `main`. Publishing requires the repository's `NPM_TOKEN` secret.
