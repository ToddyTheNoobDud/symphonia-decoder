# `@napi-rs/package-template`

## MP4 demux for NodeLink

`demuxMp4Aac(buffer)` returns raw AAC packets, decode timestamps in seconds,
and the track's AudioSpecificConfig. It requires a complete input buffer and
is intended for the Eternalbox cache.

`openMp4Audio(size, read, seekSeconds?)` opens a seekable source. Pass
`null` for a forward-only MP4 with its index before the media. The callback
must satisfy each read in full, except at EOF on a forward-only source.
Use `readPackets(limit)` for compressed AAC or Opus, and `readPcm()` for
48 kHz stereo ALAC output. Cancel pending transport reads before `close()`.

For AAC decoded by FAAD2, create one `PcmResampler(inputRate, quality)`
per stream. Feed interleaved floats and the channel count to `process()`.
It downmixes to stereo and returns signed 16-bit little-endian PCM at 48 kHz.
Collect `finish()` at end of stream to drain the resampler, or call
`close()` to discard pending frames on cancellation.

To test locally in NodeLink, build the native addon with `cargo build --release`
using a C toolchain and CMake. Install its loader, declarations and platform
binary together into NodeLink's `node_modules/@toddynnn/symphonia-decoder`.
The contract tests use FFmpeg, FFprobe and localhost HTTP. Run them from
the NodeLink checkout.

![https://github.com/napi-rs/package-template/actions](https://github.com/napi-rs/package-template/workflows/CI/badge.svg)

> Template project for writing node packages with napi-rs.

# Usage

1. Click **Use this template**.
2. **Clone** your project.
3. Run `yarn install` to install dependencies.
4. Run `yarn napi rename -n [@your-scope/package-name] -b [binary-name]` command under the project folder to rename your package.

## Install this test package

```bash
yarn add @napi-rs/package-template
```

## Ability

### Build

After `yarn build/npm run build` command, you can see `package-template.[darwin|win32|linux].node` file in project root. This is the native addon built from [lib.rs](./src/lib.rs).

### Test

With [ava](https://github.com/avajs/ava), run `yarn test/npm run test` to testing native addon. You can also switch to another testing framework if you want.

### CI

With GitHub Actions, each commit and pull request will be built and tested automatically in [`node@20`, `@node22`] x [`macOS`, `Linux`, `Windows`] matrix. You will never be afraid of the native addon broken in these platforms.

### Release

Release native package is very difficult in old days. Native packages may ask developers who use it to install `build toolchain` like `gcc/llvm`, `node-gyp` or something more.

With `GitHub actions`, we can easily prebuild a `binary` for major platforms. And with `N-API`, we should never be afraid of **ABI Compatible**.

The other problem is how to deliver prebuild `binary` to users. Downloading it in `postinstall` script is a common way that most packages do it right now. The problem with this solution is it introduced many other packages to download binary that has not been used by `runtime codes`. The other problem is some users may not easily download the binary from `GitHub/CDN` if they are behind a private network (But in most cases, they have a private NPM mirror).

In this package, we choose a better way to solve this problem. We release different `npm packages` for different platforms. And add it to `optionalDependencies` before releasing the `Major` package to npm.

`NPM` will choose which native package should download from `registry` automatically. You can see [npm](./npm) dir for details. And you can also run `yarn add @napi-rs/package-template` to see how it works.

## Develop requirements

- Install the latest `Rust`
- Install `Node.js@10+` which fully supported `Node-API`
- Install `yarn@1.x`

## Test in local

- yarn
- yarn build
- yarn test

And you will see:

```bash
$ ava --verbose

  ✔ sync function from native code
  ✔ sleep function from native code (201ms)
  ─

  2 tests passed
✨  Done in 1.12s.
```

## Release package

Ensure you have set your **NPM_TOKEN** in the `GitHub` project setting.

In `Settings -> Secrets`, add **NPM_TOKEN** into it.

Set the npm version before regenerating the loader and declarations. For `1.3.0`:

```sh
npm version 1.3.0 --no-git-tag-version --ignore-scripts
corepack yarn build
corepack yarn test
```

Review and commit the changes with `1.3.0` as the commit message. Push that
commit to `main` to run the platform builds and tests before CI publishes.
