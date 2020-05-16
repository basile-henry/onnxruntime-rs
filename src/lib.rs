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
        (API.$name.expect(&format!("ORT api: \"{}\" unavailable", stringify!($name))))($($arg),*)
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

macro_rules! ort_type {
    ($t:ident, $r:ident) => {
        pub struct $t {
            raw: *mut sys::$t,
        }

        impl Drop for $t {
            fn drop(&mut self) -> () {
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
ort_type!(ThreadingOptions, ReleaseThreadingOptions);

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

    pub fn add_custom_op_domain(self, custom_op_domain: &mut CustomOpDomain) -> Result<Self> {
        unsafe {
            checked_call!(AddCustomOpDomain, self.raw, custom_op_domain.raw)?;
        }
        Ok(self)
    }

    pub fn register_custom_ops_library<F>(
        self,
        library_path: &str,
        library_handle: F,
    ) -> Result<Self>
    where
        F: FnMut() -> () + 'static,
    {
        let library_path = CString::new(library_path)?;
        let library_handle = Box::new(Box::new(library_handle));
        unsafe {
            checked_call!(
                RegisterCustomOpsLibrary,
                self.raw,
                library_path.as_ptr(),
                Box::into_raw(library_handle) as *mut _
            )?;
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
    pub fn new(env: Env, model_path: &str, options: SessionOptions) -> Result<Self> {
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

    pub fn run_raw(
        &self,
        options: RunOptions,
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

impl Allocator {
    // TODO: Expose non default constructor?
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(GetAllocatorWithDefaultOptions, &mut raw)?;
        }
        Ok(Allocator { raw })
    }
}
