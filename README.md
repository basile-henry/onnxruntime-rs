# onnxruntime

![Cargo CI](https://github.com/basile-henry/onnxruntime-rs/workflows/Cargo%20CI/badge.svg)
![Nix CI](https://github.com/basile-henry/onnxruntime-rs/workflows/Nix%20CI/badge.svg)

This crate provides bindings to
[`onnxruntime`](https://github.com/microsoft/onnxruntime). Raw bindings
to the C API (`sys` module) as well as some opinionated wrappers to make
`onnxruntime` usable safely from plain Rust.

In order to build and use this crate with `cargo` you must either have
`onnxruntime` installed on your system or point to the release (or build) of
`onnxruntime` you want to use manually.

Tested (on Linux) with `onnxruntime` versions:
  - [`1.2.0`](https://github.com/microsoft/onnxruntime/releases/tag/v1.2.0)
  - [`1.3.0`](https://github.com/microsoft/onnxruntime/releases/tag/v1.3.0)
  - [`1.4.0`](https://github.com/microsoft/onnxruntime/releases/tag/v1.4.0)

## Download the latest release of `onnxruntime`

Download from: https://github.com/microsoft/onnxruntime/releases

Then either install it globally (for example `/usr/local/{lib,include/`) or
export the following environment variables:

`ONNXRUNTIME_LIB_DIR`: path to the `lib` directory of the release
`ONNXRUNTIME_INCLUDE_DIR`: path to the `include` directory of the release

Also make sure the `lib` directory is also part of `LD_LIBRARY_PATH`.

## Build `onnxruntime` from source

Follow the guide at:
https://github.com/microsoft/onnxruntime/blob/master/BUILD.md

For Linux/Mac OS X:
  - The lib directory will be:
    `onnxruntime/build/Linux/RelWithDebInfo/`
  - The include directory will be:
    `onnxruntime/include/onnxruntime/core/session/`

In `onnxruntime` directory:
```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel
export ONNXRUNTIME_LIB_DIR=$PWD/build/Linux/RelWithDebInfo/
export ONNXRUNTIME_INCLUDE_DIR=$PWD/include
export LD_LIBRARY_PATH=$ONNXRUNTIME_LIB_DIR:$LD_LIBRARY_PATH
```

## License

[MIT License](./LICENSE)

Copyright 2020 Basile Henry, Chris Chalmers
