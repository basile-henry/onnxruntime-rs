use std::convert::TryFrom;
use std::ffi;
use std::os::raw::c_char;
use std::ptr;

mod sys;

// Re-export enums
pub use sys::{
    ExecutionMode, GraphOptimizationLevel, ONNXTensorElementDataType, ONNXType, OrtAllocatorType,
    OrtErrorCode, OrtLoggingLevel, OrtMemType,
};

lazy_static::lazy_static! {
    static ref API: &'static sys::OrtApi = unsafe {
        let api_base = sys::OrtGetApiBase().as_ref().unwrap();
        let get_api = api_base.GetApi.unwrap();
        get_api(sys::ORT_API_VERSION).as_ref().unwrap()
    };
}

macro_rules! call {
    ($name:ident, $($arg:expr),*) => {
        (API.$name.expect(&format!("OrtApi: \"{}\" unavailable", stringify!($name))))($($arg),*)
    }
}

macro_rules! checked_call {
    ($name:ident, $($arg:expr),*) => {{
        let status = call!($name, $($arg),*);
        match OrtError::try_from(Status::from(status)) {
            Ok(err) => Err(Error::OrtError(err)),
            Err(()) => Ok(()),
        }
    }}
}

macro_rules! ort_type {
    ($t:ident, $ot:ident, $r:ident) => {
        pub struct $t {
            raw: *mut sys::$ot,
        }

        impl Drop for $t {
            fn drop(&mut self) -> () {
                unsafe { call!($r, self.raw) }
            }
        }
    };
}

ort_type!(Session, OrtSession, ReleaseSession);
ort_type!(SessionOptions, OrtSessionOptions, ReleaseSessionOptions);
ort_type!(Env, OrtEnv, ReleaseEnv);
ort_type!(Status, OrtStatus, ReleaseStatus);
ort_type!(MemoryInfo, OrtMemoryInfo, ReleaseMemoryInfo);
ort_type!(Value, OrtValue, ReleaseValue);
ort_type!(RunOptions, OrtRunOptions, ReleaseRunOptions);
ort_type!(TypeInfo, OrtTypeInfo, ReleaseTypeInfo);
ort_type!(
    TensorTypeAndShapeInfo,
    OrtTensorTypeAndShapeInfo,
    ReleaseTensorTypeAndShapeInfo
);
ort_type!(CustomOpDomain, OrtCustomOpDomain, ReleaseCustomOpDomain);
ort_type!(MapTypeInfo, OrtMapTypeInfo, ReleaseMapTypeInfo);
ort_type!(
    SequenceTypeInfo,
    OrtSequenceTypeInfo,
    ReleaseSequenceTypeInfo
);
ort_type!(ModelMetadata, OrtModelMetadata, ReleaseModelMetadata);
ort_type!(
    ThreadingOptions,
    OrtThreadingOptions,
    ReleaseThreadingOptions
);

pub struct OrtError {
    pub error_code: OrtErrorCode,
    pub error_msg: String,
}

pub enum Error {
    NulStringError(ffi::NulError),
    OrtError(OrtError),
}

type Result<T> = std::result::Result<T, Error>;

fn to_c_string(s: &str) -> Result<*const c_char> {
    Ok(ffi::CString::new(s)
        .map_err(|e| Error::NulStringError(e))?
        .as_ptr())
}

impl From<*mut sys::OrtStatus> for Status {
    fn from(raw: *mut sys::OrtStatus) -> Status {
        Status { raw }
    }
}

impl TryFrom<Status> for OrtError {
    type Error = ();

    fn try_from(status: Status) -> std::result::Result<OrtError, ()> {
        unsafe {
            if status.raw.is_null() {
                return Err(());
            }

            let error_code = call!(GetErrorCode, status.raw);
            let error_msg = ffi::CStr::from_ptr(call!(GetErrorMessage, status.raw))
                .to_string_lossy()
                .into_owned();

            Ok(OrtError {
                error_code,
                error_msg,
            })
        }
    }
}

impl Env {
    pub fn new(logging_level: OrtLoggingLevel, log_identifier: &str) -> Result<Self> {
        let log_identifier = to_c_string(log_identifier)?;
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateEnv, logging_level, log_identifier, &mut raw)?;
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

    pub fn set_optimized_model_filepath(&self, optimized_model_filepath: &str) -> Result<()> {
        let optimized_model_filepath = to_c_string(optimized_model_filepath)?;
        unsafe {
            checked_call!(
                SetOptimizedModelFilePath,
                self.raw,
                optimized_model_filepath
            )?;
        }
        Ok(())
    }
}

impl Session {
    pub fn new(env: Env, model_path: &str, options: SessionOptions) -> Result<Self> {
        let model_path = to_c_string(model_path)?;
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(CreateSession, env.raw, model_path, options.raw, &mut raw)?;
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
            .map(|n| to_c_string(n))
            .collect::<Result<Vec<_>>>()?;
        let output_names = output_names
            .iter()
            .map(|n| to_c_string(n))
            .collect::<Result<Vec<_>>>()?;
        let inputs = inputs
            .iter()
            .map(|v| v.raw as *const sys::OrtValue)
            .collect::<Vec<_>>();
        let output_size = output_names.len() as u64;
        let mut raw_outputs: *mut sys::OrtValue = ptr::null_mut();
        unsafe {
            checked_call!(
                Run,
                self.raw,
                options.raw,
                input_names.as_ptr(),
                inputs.as_ptr(),
                inputs.len() as u64,
                output_names.as_ptr(),
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

