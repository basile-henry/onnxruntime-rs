#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)] // u128 is not FFI safe

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_void, CStr, CString};
    use std::io::Result;
    use std::mem::size_of;
    use std::ptr::null_mut;
    use std::slice;

    fn check_status(api: &OrtApi, status: *mut OrtStatus) -> () {
        use std::ffi::CStr;

        let get_error_message = api.GetErrorMessage.unwrap();

        unsafe {
            if !status.is_null() {
                let msg = CStr::from_ptr(get_error_message(status));
                panic!("error: {}", msg.to_string_lossy());
            }
        }
    }

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

        let allocator: *mut OrtAllocator = {
            let get_allocator_with_default_options = api.GetAllocatorWithDefaultOptions.unwrap();
            let mut raw_allocator = null_mut();
            unsafe {
                let status = get_allocator_with_default_options(&mut raw_allocator);
                check_status(api, status);
                raw_allocator
            }
        };

        let input_name: &CStr = {
            let session_get_input_name = api.SessionGetInputName.unwrap();
            let mut raw_name = null_mut();
            unsafe {
                let status = session_get_input_name(session, 0, allocator, &mut raw_name);
                check_status(api, status);
                CStr::from_ptr(raw_name)
            }
        };

        let output_name: &CStr = {
            let session_get_output_name = api.SessionGetOutputName.unwrap();
            let mut raw_name = null_mut();
            unsafe {
                let status = session_get_output_name(session, 0, allocator, &mut raw_name);
                check_status(api, status);
                CStr::from_ptr(raw_name)
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

        let input_names = [input_name.as_ptr()].as_ptr();
        let output_names = [output_name.as_ptr()].as_ptr();

        let input_data: *mut f32 = [1., 2., 3., 4., 5., 6.].as_mut_ptr();
        let input_shape: *const i64 = [3, 2].as_ptr();

        // 3x2
        let input: *const OrtValue = {
            let create_tensor = api.CreateTensorWithDataAsOrtValue.unwrap();
            let mut raw_input = null_mut();
            unsafe {
                let status = create_tensor(
                    memory_info,
                    input_data as *mut c_void,
                    6 * size_of::<f32>() as u64, // number of bytes
                    input_shape,
                    2, // number of values
                    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
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

        let output_data: &[f32] = {
            let get_tensor_mutable_data = api.GetTensorMutableData.unwrap();
            let mut raw_output_data = null_mut();
            unsafe {
                let status = get_tensor_mutable_data(output, &mut raw_output_data);
                check_status(api, status);

                slice::from_raw_parts(raw_output_data.cast(), 3)
            }
        };

        assert_eq!(output_data, &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]);

        Ok(())
    }
}
