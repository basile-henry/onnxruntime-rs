#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
pub(crate) mod tests {

    use super::*;
    use std::ffi::{c_void, CStr, CString};
    use std::io::Result;
    use std::mem::size_of;
    use std::ptr::null_mut;
    use std::slice;

    fn check_status(api: &Api, status: *mut Status) -> () {
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
        let api_base: &ApiBase = unsafe { GetApiBase().as_ref().unwrap() };

        let api: &Api = {
            let get_api = api_base.GetApi.unwrap();
            unsafe { get_api(ORT_API_VERSION).as_ref().unwrap() }
        };

        let env: *mut Env = crate::tests::TEST_ENV.test_env.raw;

        let session_options: *mut SessionOptions = {
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

        let model_path = CString::new("testdata/matmul_1.onnx").unwrap();

        let session: *mut Session = {
            let create_session = api.CreateSession.unwrap();
            let mut raw_session = null_mut();
            unsafe {
                let status =
                    create_session(env, model_path.as_ptr(), session_options, &mut raw_session);
                check_status(api, status);
                raw_session
            }
        };

        let allocator: *mut Allocator = {
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

        let run_options: *mut RunOptions = {
            let create_run_options = api.CreateRunOptions.unwrap();
            let mut raw_run_options = null_mut();
            unsafe {
                let status = create_run_options(&mut raw_run_options);
                check_status(api, status);
                raw_run_options
            }
        };

        let memory_info: *mut MemoryInfo = {
            let create_cpu_memory_info = api.CreateCpuMemoryInfo.unwrap();
            let mut raw_memory_info = null_mut();
            unsafe {
                let status = create_cpu_memory_info(
                    AllocatorType::ArenaAllocator,
                    MemType::Cpu,
                    &mut raw_memory_info,
                );
                check_status(api, status);
                raw_memory_info
            }
        };

        let input_names = [input_name.as_ptr()];
        let output_names = [output_name.as_ptr()];

        let input_data: &mut [f32] = &mut [1., 2., 3., 4., 5., 6.];
        let input_shape: &[i64] = &[3, 2];

        // 3x2
        let input: *const Value = {
            let create_tensor = api.CreateTensorWithDataAsOrtValue.unwrap();
            let mut raw_input = null_mut();
            unsafe {
                let status = create_tensor(
                    memory_info,
                    input_data.as_mut_ptr() as *mut c_void,
                    6 * size_of::<f32>() as u64, // number of bytes
                    input_shape.as_ptr(),
                    2, // number of values
                    OnnxTensorElementDataType::Float,
                    &mut raw_input,
                );
                check_status(api, status);
                raw_input.cast()
            }
        };

        let output: *mut Value = {
            let run = api.Run.unwrap();
            let mut raw_output = null_mut();
            unsafe {
                let status = run(
                    session,
                    run_options,
                    input_names.as_ptr(),
                    &input,
                    1,
                    output_names.as_ptr(),
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

        // Cleanup
        unsafe {
            (api.ReleaseValue.unwrap())(output);
            (api.ReleaseMemoryInfo.unwrap())(memory_info);
            (api.ReleaseRunOptions.unwrap())(run_options);
            (api.ReleaseSession.unwrap())(session);
            (api.ReleaseSessionOptions.unwrap())(session_options);
        };

        Ok(())
    }
}
