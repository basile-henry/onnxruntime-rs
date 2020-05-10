use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=onnxruntime");

    let bindings = bindgen::Builder::default()
        .header("cbits/ort.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .whitelist_type("OrtApi")
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
