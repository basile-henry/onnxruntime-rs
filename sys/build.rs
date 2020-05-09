use std::env;
use std::path::PathBuf;

fn main() {
    let onnxruntime_c_api = cmake::build("../../cmake/");

    println!(
        "cargo:rustc-link-search=native={}/lib",
        onnxruntime_c_api.display()
    );
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    let header = format!(
        "{}/include/onnxruntime/core/session/onnxruntime_c_api.h",
        onnxruntime_c_api.display()
    );

    let bindings = bindgen::Builder::default()
        .header(header)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
