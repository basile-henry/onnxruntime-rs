use std::ffi::c_void;
use std::ffi::{self, CString};
use std::ptr;

mod sys;

// Re-export enums
pub use sys::{
    AllocatorType, ErrorCode, ExecutionMode, GraphOptimizationLevel, LoggingLevel, MemType,
    OnnxTensorElementDataType, OnnxType,
};

lazy_static::lazy_static! {
    static ref API: &'static sys::Api = unsafe {
        let api_base = sys::GetApiBase().as_ref().unwrap();
        let get_api = api_base.GetApi.unwrap();
        get_api(sys::ORT_API_VERSION).as_ref().unwrap()
    };
}

macro_rules! call {
    ($name:ident, $($arg:expr),*) => {
        (API.$name.expect(concat!("ORT api: \"", stringify!($name), "\" unavailable", )))($($arg),*)
    }
}

macro_rules! checked_call {
    ($name:ident, $($arg:expr),*) => {{
        let status = call!($name, $($arg),*);
        match Status::new(status) {
            Some(status) => Err(Error::OrtError(status)),
            None => Ok(()),
        }
    }}
}

// note that this be come after the macro definitions
mod value;
pub use value::Tensor;

macro_rules! ort_type {
    ($t:ident, $r:ident) => {
        pub struct $t {
            raw: *mut sys::$t,
        }

        impl Drop for $t {
            fn drop(&mut self) {
                unsafe { call!($r, self.raw) }
            }
        }
    };
}

ort_type!(Session, ReleaseSession);
ort_type!(SessionOptions, ReleaseSessionOptions);
ort_type!(Env, ReleaseEnv);
ort_type!(MemoryInfo, ReleaseMemoryInfo);
ort_type!(Value, ReleaseValue);
ort_type!(RunOptions, ReleaseRunOptions);
ort_type!(TypeInfo, ReleaseTypeInfo);
ort_type!(TensorTypeAndShapeInfo, ReleaseTensorTypeAndShapeInfo);
ort_type!(CustomOpDomain, ReleaseCustomOpDomain);
ort_type!(MapTypeInfo, ReleaseMapTypeInfo);
ort_type!(SequenceTypeInfo, ReleaseSequenceTypeInfo);
ort_type!(ModelMetadata, ReleaseModelMetadata);
// only in later versions of ort
// ort_type!(ThreadingOptions, ReleaseThreadingOptions);

#[derive(Debug)]
pub struct Status {
    pub error_code: ErrorCode,
    pub error_msg: String,
}

#[derive(Debug)]
pub enum Error {
    NulStringError(ffi::NulError),
    OrtError(Status),
}

impl From<ffi::NulError> for Error {
    fn from(e: ffi::NulError) -> Error {
        Error::NulStringError(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl Status {
    fn new(raw: *mut sys::Status) -> Option<Status> {
        unsafe {
            if raw.is_null() {
                return None;
            }

            let error_code = call!(GetErrorCode, raw);
            let error_msg = ffi::CStr::from_ptr(call!(GetErrorMessage, raw))
                .to_string_lossy()
                .into_owned();

            call!(ReleaseStatus, raw);

            Some(Status {
                error_code,
                error_msg,
            })
        }
    }
}

impl Env {
    pub fn new(logging_level: LoggingLevel, log_identifier: &str) -> Result<Self> {
        let log_identifier = CString::new(log_identifier)?;
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateEnv, logging_level, log_identifier.as_ptr(), &mut raw)?;
        }

        Ok(Env { raw })
    }
}

impl SessionOptions {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateSessionOptions, &mut raw)?;
        }

        Ok(SessionOptions { raw })
    }

    pub fn set_optimized_model_filepath(self, optimized_model_filepath: &str) -> Result<Self> {
        let optimized_model_filepath = CString::new(optimized_model_filepath)?;
        unsafe {
            checked_call!(
                SetOptimizedModelFilePath,
                self.raw,
                optimized_model_filepath.as_ptr()
            )?;
        }
        Ok(self)
    }

    pub fn enable_profiling(self, profile_file_prefix: &str) -> Result<Self> {
        let profile_file_prefix = CString::new(profile_file_prefix)?;
        unsafe {
            checked_call!(EnableProfiling, self.raw, profile_file_prefix.as_ptr())?;
        }
        Ok(self)
    }

    pub fn disable_profiling(self) -> Result<Self> {
        unsafe {
            checked_call!(DisableProfiling, self.raw)?;
        }
        Ok(self)
    }

    pub fn enable_mem_pattern(self) -> Result<Self> {
        unsafe {
            checked_call!(EnableMemPattern, self.raw)?;
        }
        Ok(self)
    }

    pub fn disable_mem_pattern(self) -> Result<Self> {
        unsafe {
            checked_call!(DisableMemPattern, self.raw)?;
        }
        Ok(self)
    }

    pub fn enable_cpu_mem_arena(self) -> Result<Self> {
        unsafe {
            checked_call!(EnableCpuMemArena, self.raw)?;
        }
        Ok(self)
    }

    pub fn disable_cpu_mem_arena(self) -> Result<Self> {
        unsafe {
            checked_call!(DisableCpuMemArena, self.raw)?;
        }
        Ok(self)
    }

    pub fn set_session_log_id(self, log_id: &str) -> Result<Self> {
        let log_id = CString::new(log_id)?;
        unsafe {
            checked_call!(SetSessionLogId, self.raw, log_id.as_ptr())?;
        }
        Ok(self)
    }

    pub fn set_session_log_verbosity_level(self, verbosity_level: i32) -> Result<Self> {
        unsafe {
            checked_call!(SetSessionLogVerbosityLevel, self.raw, verbosity_level)?;
        }
        Ok(self)
    }

    pub fn set_session_log_severity_level(self, severity_level: i32) -> Result<Self> {
        unsafe {
            checked_call!(SetSessionLogSeverityLevel, self.raw, severity_level)?;
        }
        Ok(self)
    }

    pub fn set_session_graph_optimization_level(
        self,
        graph_optimization_level: GraphOptimizationLevel,
    ) -> Result<Self> {
        unsafe {
            checked_call!(
                SetSessionGraphOptimizationLevel,
                self.raw,
                graph_optimization_level
            )?;
        }
        Ok(self)
    }

    pub fn set_intra_op_num_threads(self, intra_op_num_threads: i32) -> Result<Self> {
        unsafe {
            checked_call!(SetIntraOpNumThreads, self.raw, intra_op_num_threads)?;
        }
        Ok(self)
    }

    pub fn set_inter_op_num_threads(self, inter_op_num_threads: i32) -> Result<Self> {
        unsafe {
            checked_call!(SetInterOpNumThreads, self.raw, inter_op_num_threads)?;
        }
        Ok(self)
    }
}

impl Clone for SessionOptions {
    fn clone(&self) -> Self {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CloneSessionOptions, self.raw, &mut raw).unwrap();
        }

        SessionOptions { raw }
    }
}

impl Session {
    pub fn new(env: &Env, model_path: &str, options: &SessionOptions) -> Result<Self> {
        let model_path = CString::new(model_path)?;
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(
                CreateSession,
                env.raw,
                model_path.as_ptr(),
                options.raw,
                &mut raw
            )?;
        }

        Ok(Session { raw })
    }

    pub fn input_name(&self, ix: u64) -> Result<OrtString> {
        let alloc = Allocator::default();
        let mut raw = ptr::null_mut();

        unsafe {
            checked_call!(SessionGetInputName, self.raw, ix, alloc.raw, &mut raw)?;
        }

        Ok(OrtString { raw })
    }

    pub fn output_name(&self, ix: u64) -> Result<OrtString> {
        let alloc = Allocator::default();
        let mut raw = ptr::null_mut();

        unsafe {
            checked_call!(SessionGetOutputName, self.raw, ix, alloc.raw, &mut raw)?;
        }

        Ok(OrtString { raw })
    }

    // pub fn run(
    //     &self,
    //     options: &RunOptions,
    //     input_names: InputNames,
    //     inputs: Inputs,
    //     output_names: OutputNames,
    //     ) -> Result<Vec<Value>>
    //     where InputNames: Iterator<Item=&str>,
    //           Inputs: Iterator<Item=&Value>,
    //           OutputNames: Iterator<Item=&str>,
    // {
    // }

    pub fn run_raw(
        &self,
        options: &RunOptions,
        input_names: &[&str],
        inputs: &[&Value],
        output_names: &[&str],
    ) -> Result<Vec<Value>> {
        assert_eq!(input_names.len(), inputs.len());

        let input_names = input_names
            .iter()
            .map(|n| CString::new(*n))
            .collect::<std::result::Result<Vec<_>, ffi::NulError>>()?;
        let input_names_ptrs = input_names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        let output_names = output_names
            .iter()
            .map(|n| CString::new(*n))
            .collect::<std::result::Result<Vec<_>, ffi::NulError>>()?;
        let output_names_ptrs = output_names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        let inputs = inputs
            .iter()
            .map(|v| v.raw as *const sys::Value)
            .collect::<Vec<_>>();
        let output_size = output_names.len() as u64;
        let mut raw_outputs: *mut sys::Value = ptr::null_mut();
        unsafe {
            checked_call!(
                Run,
                self.raw,
                options.raw,
                input_names_ptrs.as_ptr(),
                inputs.as_ptr(),
                inputs.len() as u64,
                output_names_ptrs.as_ptr(),
                output_size,
                &mut raw_outputs
            )?;

            Ok(
                std::slice::from_raw_parts(&raw_outputs, output_size as usize)
                    .iter()
                    .map(|v| Value { raw: *v })
                    .collect(),
            )
        }
    }
}

pub struct Allocator {
    raw: *mut sys::Allocator,
}

impl Default for Allocator {
    fn default() -> Self {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(GetAllocatorWithDefaultOptions, &mut raw)
                .expect("GetAllocatorWithDefaultOptions");
        }
        Allocator { raw }
    }
}

impl Allocator {
    pub unsafe fn free(&self, ptr: *mut c_void) {
        checked_call!(AllocatorFree, self.raw, ptr).expect("AllocatorFree");
    }
}

use std::ffi::CStr;
use std::os::raw::c_char;

/// An ort string with the default allocator
pub struct OrtString {
    raw: *const c_char,
}

impl std::ops::Deref for OrtString {
    type Target = CStr;

    fn deref(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.raw) }
    }
}

impl Drop for OrtString {
    fn drop(&mut self) {
        let alloc = Allocator::default();
        unsafe { alloc.free(self.raw as _) }
    }
}

impl MemoryInfo {
    pub fn cpu_memory_info(alloc_type: AllocatorType, mem_type: MemType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateCpuMemoryInfo, alloc_type, mem_type, &mut raw)?;
        }

        Ok(MemoryInfo { raw })
    }
}

impl RunOptions {
    pub fn new() -> RunOptions {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateRunOptions, &mut raw).expect("CreateRunOptions");
        }

        RunOptions { raw }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_mul() -> Result<()> {
        let env = Env::new(LoggingLevel::Warning, "test")?;

        let so = SessionOptions::new()?;

        let model_path = "../onnxruntime/test/testdata/matmul_1.onnx";
        let session = Session::new(&env, model_path, &so)?;
        let in_name = session.input_name(0)?;
        let out_name = session.output_name(0)?;

        let ro = RunOptions::new();

        let mem_info = MemoryInfo::cpu_memory_info(AllocatorType::ArenaAllocator, MemType::Cpu)?;

        let input_names = vec![in_name.to_str().unwrap()];
        let output_names = vec![out_name.to_str().unwrap()];

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_shape = vec![3, 2];

        let input_tensor = Tensor::new(mem_info, input_shape, input_data)?;

        let output = session.run_raw(&ro, &input_names, &[input_tensor.value()], &output_names)?;

        let output_value = output.into_iter().next().unwrap();
        let output_tensor = output_value.as_tensor::<f32>().ok().expect("as_tensor");

        assert_eq!(
            &output_tensor[..],
            &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]
        );

        Ok(())
    }
}
