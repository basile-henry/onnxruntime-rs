use std::ffi::{self, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

pub mod sys;
// Re-export enums
pub use sys::{
    AllocatorType, ErrorCode, ExecutionMode, GraphOptimizationLevel, LoggingLevel, MemType,
    OnnxTensorElementDataType, OnnxType,
};

#[macro_use]
mod api;

mod allocator;
pub use allocator::Allocator;

// note that this be come after the macro definitions (in api)
mod value;
pub use value::{OrtType, Tensor, Val};

macro_rules! ort_type {
    ($t:ident, $r:ident) => {
        pub struct $t {
            raw: *mut sys::$t,
        }

        impl $t {
            pub fn raw(&self) -> *mut sys::$t {
                self.raw
            }
            pub unsafe fn from_raw(raw: *mut sys::$t) -> Self {
                Self { raw }
            }
        }

        impl Drop for $t {
            fn drop(&mut self) {
                call!(@unsafe @raw $r, self.raw)
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

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NulStringError(_) => write!(fmt, "null string error"),
            Error::OrtError(st) => write!(fmt, "{:?}: {}", st.error_code, st.error_msg),
        }
    }
}

impl From<ffi::NulError> for Error {
    fn from(e: ffi::NulError) -> Error {
        Error::NulStringError(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl Status {
    fn new(raw: *mut sys::Status) -> Option<Status> {
        if raw.is_null() {
            return None;
        }

        let error_code = call!(@unsafe @raw GetErrorCode, raw);
        let error_msg = unsafe {
            CStr::from_ptr(call!(@raw GetErrorMessage, raw))
                .to_string_lossy()
                .into_owned()
        };

        call!(@unsafe @raw ReleaseStatus, raw);

        Some(Status {
            error_code,
            error_msg,
        })
    }
}

impl Env {
    pub fn new(logging_level: LoggingLevel, log_identifier: &str) -> Result<Self> {
        let log_identifier = CString::new(log_identifier)?;
        let raw = call!(@unsafe @ptr CreateEnv, logging_level, log_identifier.as_ptr())?;
        Ok(Env { raw })
    }
}

macro_rules! option {
    ($(#[$outer:meta])* $ort_name:ident => $name:ident) => {
        $(#[$outer])*
        pub fn $name(&mut self) -> Result<&mut Self> {
            call!(@unsafe $ort_name, self.raw)?;
            Ok(self)
        }
    };

    ($(#[$outer:meta])* $ort_name:ident => $name:ident($arg_name:ident: $arg_ty:ty $(| .$fn:ident())?)) => {
        $(#[$outer])*
        pub fn $name(&mut self, $arg_name: $arg_ty) -> Result<&mut Self> {
            call!(@unsafe $ort_name, self.raw, $arg_name$(.$fn())?)?;
            Ok(self)
        }
    };
}

impl SessionOptions {
    pub fn new() -> Result<Self> {
        let raw = call!(@unsafe @ptr CreateSessionOptions)?;
        Ok(SessionOptions { raw })
    }

    pub fn set_optimized_model_filepath(
        &mut self,
        optimized_model_filepath: &str,
    ) -> Result<&mut Self> {
        let optimized_model_filepath = CString::new(optimized_model_filepath)?;
        call!(@unsafe
            SetOptimizedModelFilePath,
            self.raw,
            optimized_model_filepath.as_ptr()
        )?;
        Ok(self)
    }

    pub fn enable_profiling(&mut self, profile_file_prefix: &str) -> Result<&mut Self> {
        let profile_file_prefix = CString::new(profile_file_prefix)?;
        call!(@unsafe EnableProfiling, self.raw, profile_file_prefix.as_ptr())?;
        Ok(self)
    }

    option!(
        /// Disables profiling.
        DisableProfiling => disable_profiling
    );
    option!(EnableMemPattern => enable_mem_pattern);
    option!(DisableMemPattern => disable_mem_pattern);
    option!(EnableCpuMemArena => enable_cpu_mem_arena);
    option!(DisableCpuMemArena => disable_cpu_mem_arena);

    pub fn set_session_log_id(&mut self, log_id: &str) -> Result<&mut Self> {
        let log_id = CString::new(log_id)?;
        call!(@unsafe SetSessionLogId, self.raw, log_id.as_ptr())?;
        Ok(self)
    }

    option!(EnableProfiling => en_prof(path: &CStr | .as_ptr()));
    option!(SetSessionLogVerbosityLevel => set_session_log_verbosity_level(verbosity_level: i32));
    option!(SetSessionLogSeverityLevel => set_session_log_severity_level(severity_level: i32));
    option!(SetSessionGraphOptimizationLevel =>
        set_session_graph_optimization_level(graph_optimization_level: GraphOptimizationLevel));
    option!(SetIntraOpNumThreads => set_intra_op_num_threads(intra_op_num_threads: i32));
    option!(SetInterOpNumThreads => set_inter_op_num_threads(intra_op_num_threads: i32));
}

impl Clone for SessionOptions {
    fn clone(&self) -> Self {
        let raw = call!(@unsafe @ptr @expect CloneSessionOptions, self.raw);
        SessionOptions { raw }
    }
}

impl Session {
    pub fn new(env: &Env, model_path: &str, options: &SessionOptions) -> Result<Self> {
        let model_path = CString::new(model_path)?;
        let raw = call!(@unsafe @ptr CreateSession, env.raw, model_path.as_ptr(), options.raw)?;
        Ok(Session { raw })
    }

    pub fn input_count(&self) -> usize {
        call!(@unsafe @int @expect SessionGetInputCount, self.raw) as usize
    }

    pub fn output_count(&self) -> usize {
        call!(@unsafe @int @expect SessionGetOutputCount, self.raw) as usize
    }

    pub fn overridable_initializer_count(&self) -> usize {
        call!(@unsafe @int @expect SessionGetOverridableInitializerCount, self.raw) as usize
    }

    pub fn input_name(&self, ix: u64) -> Result<OrtString> {
        let alloc = Allocator::default();
        let raw = call!(@unsafe @ptr SessionGetInputName, self.raw, ix, alloc.as_ptr())?;
        Ok(OrtString { raw })
    }

    pub fn output_name(&self, ix: u64) -> Result<OrtString> {
        let alloc = Allocator::default();
        let raw = call!(@unsafe @ptr SessionGetOutputName, self.raw, ix, alloc.as_ptr())?;
        Ok(OrtString { raw })
    }

    pub fn run_mut(
        &self,
        options: &RunOptions,
        input_names: &[&CStr],
        inputs: &[&Val],
        output_names: &[&CStr],
        outputs: &mut [&mut Val],
    ) -> Result<()> {
        assert_eq!(input_names.len(), inputs.len());
        assert_eq!(output_names.len(), outputs.len());

        call!(@unsafe
            Run,
            self.raw,
            options.raw,
            input_names.as_ptr() as *const *const c_char,
            inputs.as_ptr() as *const *const sys::Value,
            inputs.len() as u64,
            output_names.as_ptr() as *const *const c_char,
            output_names.len() as u64,
            outputs.as_mut_ptr() as *mut *mut sys::Value
        )
    }

    pub fn run_raw(
        &self,
        options: &RunOptions,
        input_names: &[&CStr],
        inputs: &[&Val],
        output_names: &[&CStr],
    ) -> Result<Vec<Value>> {
        assert_eq!(input_names.len(), inputs.len());

        let output_size = output_names.len() as u64;
        let mut raw_outputs: *mut sys::Value = ptr::null_mut();
        call!(@unsafe
            Run,
            self.raw,
            options.raw,
            input_names.as_ptr() as *const *const c_char,
            inputs.as_ptr() as *const *const sys::Value,
            inputs.len() as u64,
            output_names.as_ptr() as *const *const c_char,
            output_size,
            &mut raw_outputs
        )?;

        unsafe {
            Ok(
                std::slice::from_raw_parts(&raw_outputs, output_size as usize)
                    .iter()
                    .map(|v| Value { raw: *v })
                    .collect(),
            )
        }
    }
}

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
        let raw = call!(@unsafe @ptr CreateCpuMemoryInfo, alloc_type, mem_type)?;
        Ok(MemoryInfo { raw })
    }
}

unsafe impl Sync for MemoryInfo {}
unsafe impl Send for MemoryInfo {}

lazy_static::lazy_static! {
    /// The standard cpu memory info.
    pub static ref CPU_ARENA: MemoryInfo = {
        MemoryInfo::cpu_memory_info(AllocatorType::ArenaAllocator, MemType::Cpu).expect("CPU_ARENA")
    };
}

impl RunOptions {
    pub fn new() -> RunOptions {
        let raw = call!(@unsafe @ptr @expect CreateRunOptions);
        RunOptions { raw }
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! unexpected {
    () => {};
}

#[macro_export]
/// A macro for calling run on a session. Usage:
///
/// ```ignore
/// run!(my_session(&my_session_options) =>
///    my_input_name: &my_input_tensor,
///    my_output_name: &mut my_output_tensor,
/// )?;
/// ```
/// Trailing commas are required.
macro_rules! run {
    // finished with a trailing comma
    ( @map $session:ident $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]) => {
        $session.run_mut(
                    $ro,
                    &[$($in_names,)*],
                    &[$($in_vals,)*],
                    &[$($out_names,)*],
                    &mut [$($out_vals,)*],
                )
    };

    // &mut reference means output
    ( @map $session:ident $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
            $name:ident: &mut $val:expr, $($rest:tt)* ) => {
        run!(@map $session $ro;
                    [$($in_names,)*] [$($in_vals,)*]
                    [$($out_names,)* $name.as_ref(),] [$($out_vals,)* $val.as_mut(),]
                    $($rest)*
        )
    };

    // &reference means input
    ( @map $session:ident $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
        $name:ident: &$val:expr, $($rest:tt)* ) => {
        run!(@map $session $ro;
                    [$($in_names,)* $name.as_ref(),] [$($in_vals,)* $val.as_ref(),]
                    [$($out_names,)*] [$($out_vals,)*]
                    $($rest)*
        )
    };

    // something didn't match for the io mapping (doing this prevents more permissive matches from
    // matching the @map and giving confusing type errors).
    ( @map $($tt:tt)+ ) => { unexpected!($($tt)+) };

    ( $session:ident($ro:expr) => $($tt:tt)+ ) => { run!(@map $session $ro; [] [] [] [] $($tt)+) };
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct TestEnv {
        pub(crate) test_env: Env,
    }

    unsafe impl Send for TestEnv {}
    unsafe impl Sync for TestEnv {}

    lazy_static::lazy_static! {
        /// There should only be one ORT environment existing at any given time.
        /// This test environment is intended to be used by all the tests that
        /// need an ORT environment instead of each creating their own.
        pub(crate) static ref TEST_ENV: TestEnv = TestEnv {
            test_env: Env::new(LoggingLevel::Fatal, "test").unwrap()
        };
    }

    #[test]
    fn mat_mul() -> Result<()> {
        let env = &TEST_ENV.test_env;

        let so = SessionOptions::new()?;

        let model_path = "testdata/matmul_1.onnx";
        let session = Session::new(&env, model_path, &so)?;
        let in_name = session.input_name(0)?;
        let out_name = session.output_name(0)?;

        let ro = RunOptions::new();

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_shape = vec![3, 2];
        let input_tensor = Tensor::new(input_shape, input_data)?;

        // immutable version
        let output = session.run_raw(&ro, &[&in_name], &[input_tensor.value()], &[&out_name])?;

        let output_value = output.into_iter().next().unwrap();
        let output_tensor = output_value.as_tensor::<f32>().ok().expect("as_tensor");

        assert_eq!(
            &output_tensor[..],
            &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]
        );

        // mutable version
        let mut output_tensor = Tensor::<f32>::init(vec![3, 1], 0.0)?;

        run!(session(&ro) =>
            in_name: &input_tensor,
            out_name: &mut output_tensor,
        )?;

        assert_eq!(
            &output_tensor[..],
            &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]
        );

        Ok(())
    }
}
