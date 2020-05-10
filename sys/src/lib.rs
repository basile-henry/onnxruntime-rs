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
    use std::ffi::{c_void, CStr, CString};
    use std::io::Result;
    use std::mem::size_of;
    use std::ptr::null_mut;

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
            let mut raw_env = null_mut();
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
            let mut raw_session_options = null_mut();
            unsafe {
                let status = create_session_options(&mut raw_session_options);
                check_status(api, status);
                set_intra_op_num_threads(raw_session_options, 1);
            }
            raw_session_options
        };

        let model_path = CString::new("../../onnxruntime/test/testdata/matmul_1.onnx").unwrap();

        let session: *mut OrtSession = {
            let create_session = api.CreateSession.unwrap();
            let mut raw_session = null_mut();
            unsafe {
                let status =
                    create_session(env, model_path.as_ptr(), session_options, &mut raw_session);
                check_status(api, status);
                raw_session
            }
        };

        let run_options: *mut OrtRunOptions = {
            let create_run_options = api.CreateRunOptions.unwrap();
            let mut raw_run_options = null_mut();
            unsafe {
                let status = create_run_options(&mut raw_run_options);
                check_status(api, status);
                raw_run_options
            }
        };

        let memory_info: *mut OrtMemoryInfo = {
            let create_cpu_memory_info = api.CreateCpuMemoryInfo.unwrap();
            let mut raw_memory_info = null_mut();
            unsafe {
                let status = create_cpu_memory_info(
                    OrtAllocatorType_OrtArenaAllocator,
                    OrtMemType_OrtMemTypeCPU,
                    &mut raw_memory_info,
                );
                check_status(api, status);
                raw_memory_info
            }
        };

        let input_names = [CString::new("X").unwrap().as_ptr()].as_ptr();
        let output_names = [CString::new("Y").unwrap().as_ptr()].as_ptr();

        let input_data: *mut f64 = [1., 2., 3., 4., 5., 6.].as_mut_ptr();
        let input_shape: *const i64 = [3, 2].as_ptr();

        // 3x2
        let input: *const OrtValue = {
            let create_tensor = api.CreateTensorWithDataAsOrtValue.unwrap();
            let mut raw_input = null_mut();
            unsafe {
                let status = create_tensor(
                    memory_info,
                    input_data as *mut c_void,
                    6 * size_of::<f64>() as u64,
                    input_shape,
                    2,
                    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                    &mut raw_input,
                );
                check_status(api, status);
                raw_input.cast()
            }
        };

        let output: *mut OrtValue = {
            let run = api.Run.unwrap();
            let mut raw_output = null_mut();
            unsafe {
                let status = run(
                    session,
                    run_options,
                    input_names,
                    &input,
                    1,
                    output_names,
                    1,
                    &mut raw_output,
                );
                check_status(api, status);
                raw_output
            }
        };

        Ok(())
    }
}
