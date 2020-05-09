#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)] // u128 is not FFI safe

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Result;
    use std::mem::MaybeUninit;

    #[test]
    fn simple_inference() -> Result<()> {
        let mat_mul = "../../../onnxruntime/test/testdata/matmul_1.onnx";

        let api_base: &OrtApiBase = unsafe { OrtGetApiBase().as_ref().unwrap() };

        let api: &OrtApi = unsafe {
            (api_base.GetApi.unwrap())(ORT_API_VERSION)
                .as_ref()
                .unwrap()
        };

        let env: OrtEnv = {
            let name = CString::new("test").unwrap().as_ptr();
            let create_env = api.CreateEnv.unwrap();
            unsafe {
                let mut raw_env = MaybeUninit::uninit();
                create_env(
                    OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                    name,
                    &mut raw_env.as_mut_ptr(),
                );
                raw_env.assume_init()
            }
        };

        // OrtGetApiBase().GetApi(ORT_API_VERSION);

        // let env = OrtApi.CreateEnv;

        Ok(())
    }
}
