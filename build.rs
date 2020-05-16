use bindgen::callbacks::{EnumVariantValue, ParseCallbacks};
use heck::CamelCase;
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
        }
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
        .blacklist_type("__int64_t")
        .blacklist_type("__uint32_t")
        .rustified_non_exhaustive_enum("*")
        .parse_callbacks(Box::new(CustomEnums))
        .no_copy(".*")
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[derive(Debug)]
struct CustomEnums;

impl ParseCallbacks for CustomEnums {
    fn item_name(&self, name: &str) -> Option<String> {
        if name.starts_with("Ort") {
            return Some(name.replace("Ort", ""));
        }

        if name.starts_with("ONNX") {
            return Some(name.replace("ONNX", "Onnx"));
        }

        None
    }

    fn enum_variant_name(
        &self,
        enum_name: Option<&str>,
        variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        let mut variant_name = variant_name.to_camel_case();

        if let Some(enum_name) = enum_name {
            let enum_name = enum_name.replace("enum ", "").to_camel_case();

            if variant_name.starts_with(&enum_name) {
                variant_name = variant_name.replace(&enum_name, "");
            }
        }

        if variant_name.starts_with("Ort") {
            variant_name = variant_name.replace("Ort", "");
        }

        //Special case BFloat16
        variant_name = variant_name.replace("Bfloat16", "BFloat16");

        Some(variant_name)
    }
}
