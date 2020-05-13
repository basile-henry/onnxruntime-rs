# onnxruntime

This crate contains bindings to onnxruntime.

In order to build/use this crate you must either have onnxruntime installed on
your system or build the onnxruntime project from source.

To build from source (Linux) do the following from the root of the repo:

```bash
./build.sh --config RelWithDebInfo --build_shared_lib --parallel
export ONNXRUNTIME_LIB_DIR=$PWD/build/Linux/RelWithDebInfo/
export ONNXRUNTIME_INCLUDE_DIR=$PWD/include/onnxruntime/core/session/
export LD_LIBRARY_PATH=$ONNXRUNTIME_LIB_DIR:$LD_LIBRARY_PATH
```

