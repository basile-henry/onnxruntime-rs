use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ONNXRUNTIME_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ONNXRUNTIME_INCLUDE_DIR");

    match std::env::var("ONNXRUNTIME_LIB_DIR") {
        Ok(path) => println!("cargo:rustc-link-search={}", path),
        Err(_) => (),
    };

    let mut clang_args = String::new();
    match std::env::var("ONNXRUNTIME_INCLUDE_DIR") {
        Ok(path) => {
            clang_args = format!("-I{}", path);
        },
        Err(_) => (),
    };

    println!("cargo:rustc-link-lib=onnxruntime");

    let bindings = bindgen::Builder::default()
        .header("cbits/ort.h")
        .clang_arg(clang_args)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .whitelist_function("OrtGetApiBase")
        .whitelist_var("ORT_.*")
        .whitelist_recursively(true)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
