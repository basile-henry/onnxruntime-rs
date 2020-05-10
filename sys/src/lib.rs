#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)] // u128 is not FFI safe

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn check_status(api: &OrtApi, status: *mut OrtStatus) -> () {
    use std::ffi::CStr;


    let get_error_message = api.GetErrorMessage.unwrap();
    // let release_status = api.ReleaseStatus.unwrap();

    unsafe {
        if !status.is_null() {

            let msg = CStr::from_ptr(get_error_message(status));
            panic!("error: {}", msg.to_string_lossy());
            // release_status(status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ffi::CString;
    use std::io::Result;

    #[test]
    fn mat_mul() -> Result<()> {

        let api_base: &OrtApiBase = unsafe { OrtGetApiBase().as_ref().unwrap() };

        let api: &OrtApi = {
            let get_api = api_base.GetApi.unwrap();
            unsafe { get_api(ORT_API_VERSION).as_ref().unwrap() }
        };

        let env: *mut OrtEnv = {
            let name = CStr::from_bytes_with_nul(b"test\0").unwrap().as_ptr();
            let create_env = api.CreateEnv.unwrap();
            let mut raw_env = std::ptr::null_mut();
            unsafe {
                let status = create_env(
                    OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                    name,
                    &mut raw_env,
                );
                check_status(api, status);
                raw_env
            }
        };

        let session_options: *mut OrtSessionOptions = {
            let create_session_options = api.CreateSessionOptions.unwrap();
            let set_intra_op_num_threads = api.SetIntraOpNumThreads.unwrap();
            let mut raw_session_options = std::ptr::null_mut();
            unsafe {
                let status = create_session_options(&mut raw_session_options);
                check_status(api, status);
                set_intra_op_num_threads(raw_session_options, 1);
            }
            raw_session_options
        };

        let model_path = CString::new("../../onnxruntime/test/testdata/matmul_1.onnx")
            .unwrap();

        let _session: *mut OrtSession = {
            let create_session = api.CreateSession.unwrap();
            let mut raw_session = std::ptr::null_mut();
            unsafe {
                let status = create_session(
                    env,
                    model_path.as_ptr(),
                    session_options,
                    &mut raw_session,
                );
                check_status(api, status);
                raw_session
            }
        };

        Ok(())
    }
}
