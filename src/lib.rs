use std::cell::RefCell;
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
pub use value::{OrtType, Tensor, TensorInfo, TensorView, TensorViewMut, Val};

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
    pub unsafe fn from_raw(raw: *mut sys::Status) -> Option<Status> {
        if raw.is_null() {
            return None;
        }

        let error_code = call!(@raw GetErrorCode, raw);
        let error_msg = CStr::from_ptr(call!(@raw GetErrorMessage, raw))
            .to_string_lossy()
            .into_owned();

        call!(@raw ReleaseStatus, raw);

        Some(Status {
            error_code,
            error_msg,
        })
    }
}

unsafe impl Send for Env {}
unsafe impl Sync for Env {}

impl Env {
    pub fn new(logging_level: LoggingLevel, log_identifier: &str) -> Result<Self> {
        let log_identifier = CString::new(log_identifier)?;
        let raw = call!(@unsafe @ptr CreateEnv, logging_level, log_identifier.as_ptr())?;
        Ok(Env { raw })
    }
}

macro_rules! options {
    () => {};
    ($(#[$outer:meta])* fn $name:ident() { $ort_name:ident }; $($rest:tt)*) => {
        $(#[$outer])*
        pub fn $name(&mut self) -> Result<&mut Self> {
            call!(@unsafe $ort_name, self.raw)?;
            Ok(self)
        }
        options! {$($rest)*}
    };

    // treat &str specially because we make a cstring version
    ($(#[$outer:meta])* fn $name:ident($arg_name:ident: &str) { $ort_name:ident }; $($rest:tt)*) => {
        $(#[$outer])*
        pub fn $name(&mut self, $arg_name: &str) -> Result<&mut Self> {
            let cstr = CString::new($arg_name)?;
            call!(@unsafe $ort_name, self.raw, cstr.as_ptr())?;
            Ok(self)
        }
        options! {$($rest)*}
    };

    ($(#[$outer:meta])*
     fn $name:ident($arg_name:ident: $arg_ty:ty $(| .$fn:ident())?) {$ort_name:ident};
     $($rest:tt)*) => {
        $(#[$outer])*
        pub fn $name(&mut self, $arg_name: $arg_ty) -> Result<&mut Self> {
            call!(@unsafe $ort_name, self.raw, $arg_name$(.$fn())?)?;
            Ok(self)
        }
        options! {$($rest)*}
    };
}

impl SessionOptions {
    pub fn new() -> Result<Self> {
        let raw = call!(@unsafe @ptr CreateSessionOptions)?;
        Ok(SessionOptions { raw })
    }

    options! {
    fn enable_mem_pattern() { EnableMemPattern };
    fn disable_mem_pattern() { DisableMemPattern };
    fn set_optimized_model_filepath(optimized_model_filepath: &str) { SetOptimizedModelFilePath };
    fn enable_profiling(profile_file_prefix: &str) { EnableProfiling };
    fn disable_profiling() { DisableProfiling };
    fn enable_cpu_mem_arena() { EnableCpuMemArena };
    fn disable_cpu_mem_arena() { DisableCpuMemArena };
    fn set_session_log_id(log_id: &str) { SetSessionLogId };
    fn en_prof(path: &CStr | .as_ptr()) { EnableProfiling };
    fn set_session_log_verbosity_level(verbosity_level: i32) { SetSessionLogVerbosityLevel };
    fn set_session_log_severity_level(severity_level: i32) { SetSessionLogSeverityLevel };
    fn set_session_graph_optimization_level(graph_optimization_level: GraphOptimizationLevel)
        { SetSessionGraphOptimizationLevel };
    fn set_intra_op_num_threads(intra_op_num_threads: i32) { SetIntraOpNumThreads };
    fn set_inter_op_num_threads(intra_op_num_threads: i32) { SetInterOpNumThreads };
    }
}

impl TypeInfo {
    pub fn onnx_type(&self) -> OnnxType {
        call!(@unsafe @arg OnnxType::Unknown; @expect GetOnnxTypeFromTypeInfo, self.raw)
    }

    pub fn tensor_info(&self) -> Option<&TensorInfo> {
        let raw = call!(@unsafe @arg ptr::null(); @expect CastTypeInfoToTensorInfo, self.raw);
        if raw.is_null() {
            None
        } else {
            Some(unsafe { TensorInfo::from_raw(raw) })
        }
    }
}

impl Clone for SessionOptions {
    fn clone(&self) -> Self {
        let raw = call!(@unsafe @ptr @expect CloneSessionOptions, self.raw);
        SessionOptions { raw }
    }
}

#[derive(Clone, Copy)]
enum ArgType {
    Input,
    Output,
    Initialiser,
}

/// Iterator over some of the arguments of a session. Either `inputs`, `outputs` or
/// `overridable_initializers`.
pub struct Arguments<'a> {
    session: &'a Session,
    ix: usize,
    num_args: usize,
    arg_type: ArgType,
}

impl<'a> Iterator for Arguments<'a> {
    type Item = ArgumentInfo<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.num_args {
            None
        } else {
            let info = self.session.argument(self.ix, self.arg_type);
            self.ix += 1;
            Some(info)
        }
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.ix = std::cmp::min(self.num_args, self.ix + n);
        self.next()
    }
}

impl<'a> ExactSizeIterator for Arguments<'a> {
    fn len(&self) -> usize {
        self.ix - self.num_args
    }
}

/// Infomation about a particular argument.
pub struct ArgumentInfo<'a> {
    session: &'a Session,
    ix: usize,
    arg_type: ArgType,
    info: RefCell<Option<TypeInfo>>,
}

impl<'a> ArgumentInfo<'a> {
    /// The name of this argument
    pub fn name(&self) -> OrtString {
        let alloc = Allocator::default();
        let sess = self.session.raw;
        let ix = self.ix as u64;
        let raw = match self.arg_type {
            ArgType::Input => {
                call!(@unsafe @ptr @expect SessionGetInputName, sess, ix, alloc.as_ptr())
            }
            ArgType::Output => {
                call!(@unsafe @ptr @expect SessionGetOutputName, sess, ix, alloc.as_ptr())
            }
            ArgType::Initialiser => {
                call!(@unsafe @ptr @expect SessionGetOverridableInitializerName, sess, ix, alloc.as_ptr())
            }
        };
        OrtString { raw }
    }

    /// The index of this argument
    pub fn index(&self) -> usize {
        self.ix
    }

    /// lazy `TypeInfo` instantiation.
    unsafe fn type_info(&self) -> &TypeInfo {
        match self.info.try_borrow_unguarded().expect("arg_info") {
            Some(info) => info,
            None => {
                let sess = self.session.raw;
                let ix = self.ix as u64;
                let raw = match self.arg_type {
                    ArgType::Input => call!(@ptr @expect SessionGetInputTypeInfo, sess, ix),
                    ArgType::Output => call!(@ptr @expect SessionGetOutputTypeInfo, sess, ix),
                    ArgType::Initialiser => {
                        call!(@ptr @expect SessionGetOverridableInitializerTypeInfo, sess, ix)
                    }
                };
                let type_info = TypeInfo { raw };
                self.info.replace(Some(type_info));
                self.info
                    .try_borrow_unguarded()
                    .expect("arg_info")
                    .as_ref()
                    .unwrap()
            }
        }
    }

    /// The type of this argument.
    pub fn onnx_type(&self) -> OnnxType {
        unsafe { self.type_info().onnx_type() }
    }

    /// `true` if this argument is a `Tensor` or a `Sparsetensor`.
    pub fn is_tensor(&self) -> bool {
        matches!(self.onnx_type(), OnnxType::Tensor | OnnxType::Sparsetensor)
    }

    /// Info about this tensor (like dimensions).
    pub fn tensor_info(&self) -> Option<&TensorInfo> {
        unsafe { self.type_info().tensor_info() }
    }
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn new(env: &Env, model_path: &str, options: &SessionOptions) -> Result<Self> {
        let model_path = CString::new(model_path)?;
        let raw = call!(@unsafe @ptr CreateSession, env.raw, model_path.as_ptr(), options.raw)?;
        Ok(Session { raw })
    }

    fn argument(&self, ix: usize, arg_type: ArgType) -> ArgumentInfo {
        ArgumentInfo {
            session: &self,
            ix,
            arg_type: arg_type,
            info: RefCell::new(None),
        }
    }

    fn arguments(&self, arg_type: ArgType) -> Arguments {
        Arguments {
            session: self,
            ix: 0,
            num_args: self.arg_count(arg_type),
            arg_type,
        }
    }

    fn arg_count(&self, arg_type: ArgType) -> usize {
        match arg_type {
            ArgType::Input => call!(@unsafe @int @expect SessionGetInputCount, self.raw) as usize,
            ArgType::Output => call!(@unsafe @int @expect SessionGetOutputCount, self.raw) as usize,
            ArgType::Initialiser => {
                call!(@unsafe @int @expect SessionGetOverridableInitializerCount, self.raw) as usize
            }
        }
    }

    /// Gets the input with the given index. Will error if the index is out of bounds.
    pub fn input(&self, ix: usize) -> ArgumentInfo {
        self.argument(ix, ArgType::Input)
    }

    /// Gets the output with the given index. Will error if the index is out of bounds.
    pub fn output(&self, ix: usize) -> ArgumentInfo {
        self.argument(ix, ArgType::Output)
    }

    /// Gets the overridable initializer with the given index. Will error if the index is out of bounds.
    pub fn overridable_initializer(&self, ix: usize) -> ArgumentInfo {
        self.argument(ix, ArgType::Initialiser)
    }

    /// Gets an iterator over the inputs of this session.
    pub fn inputs(&self) -> Arguments {
        self.arguments(ArgType::Input)
    }

    /// Gets an iterator over the outputs of this session.
    pub fn outputs(&self) -> Arguments {
        self.arguments(ArgType::Output)
    }

    /// Gets an iterator over the overridable initializers of this session.
    pub fn overridable_initializers(&self) -> Arguments {
        self.arguments(ArgType::Initialiser)
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
            cstr_ptrs(input_names).as_ptr(),
            inputs.as_ptr() as *const *const sys::Value,
            inputs.len() as u64,
            cstr_ptrs(output_names).as_ptr(),
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
            cstr_ptrs(input_names).as_ptr(),
            inputs.as_ptr() as *const *const sys::Value,
            inputs.len() as u64,
            cstr_ptrs(output_names).as_ptr(),
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

/// Note that this can't be replaced by a cast because internally `CStr` is represented by a slice.
fn cstr_ptrs(slice: &[&CStr]) -> Vec<*const c_char> {
    slice.iter().map(|cstr| cstr.as_ptr()).collect()
}

/// An ort string with the default allocator
pub struct OrtString {
    raw: *const c_char,
}

unsafe impl Send for OrtString {}
unsafe impl Sync for OrtString {}

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

    options! {
    /// Set a flag so that ALL incomplete OrtRun calls that are using this instance of
    /// `RunOptions` will exit as soon as possible.
    fn set_terminate() {RunOptionsSetTerminate};

    /// Unset the terminate flag to enable this `RunOptions` instance being used in new `run`
    /// calls.
    fn unset_terminate() {RunOptionsUnsetTerminate};

    fn set_log_verbosity_level(verbositity_level: i32) {RunOptionsSetRunLogVerbosityLevel};
    fn set_log_severity_level(severity_level: i32) {RunOptionsSetRunLogSeverityLevel};
    fn set_tag(verbositity_level: &str) {RunOptionsSetRunTag};
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
    // finished, call run_mut with the args
    ( @map $session:expr; $ro:expr;
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

    ( @to_name $name:ident ) => {{
        $name.as_ref()
    }};

    ( @to_name $name:literal ) => {
        ::std::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()).unwrap()
    };

    // &reference means input
    ( @map $session:expr; $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
        $name:tt: &$val:expr, $($rest:tt)* ) => {
        run!(@map $session; $ro;
                    [$($in_names,)* run!(@to_name $name),] [$($in_vals,)* $val.as_ref(),]
                    [$($out_names,)*] [$($out_vals,)*]
                    $($rest)*
        )
    };

    // &mut reference means output
    ( @map $session:expr; $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
            $name:tt: &mut $val:expr, $($rest:tt)* ) => {
        run!(@map $session; $ro;
                    [$($in_names,)*] [$($in_vals,)*]
                    [$($out_names,)* run!(@to_name $name),] [$($out_vals,)* $val.as_mut(),]
                    $($rest)*
        )
    };

    // handle no trailing comma (supper annoying, need & and &mut case)
    ( @map $session:expr; $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
            $name:tt: & $val:expr ) => {
        run!(@map $session; $ro; [$($in_names,)*] [$($in_vals,)*] [$($out_names,)*] [$($out_vals,)*] $name: &mut $val,)
    };
    ( @map $session:expr; $ro:expr;
            [$($in_names:expr,)*] [$($in_vals:expr,)*]
            [$($out_names:expr,)*] [$($out_vals:expr,)*]
            $name:tt: &mut $val:expr ) => {
        run!(@map $session; $ro; [$($in_names,)*] [$($in_vals,)*] [$($out_names,)*] [$($out_vals,)*] $name: &mut $val,)
    };

    // something didn't match for the io mapping (doing this prevents more permissive matches from
    // matching the @map and giving confusing type errors).
    ( @map $($tt:tt)+ ) => { unexpected!($($tt)+) };

    ( $session:expr, $ro:expr, $($tt:tt)+ ) => { run!(@map $session; $ro; [] [] [] [] $($tt)+) };
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct TestEnv {
        pub(crate) test_env: Env,
    }

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
        let in_name = session.input(0).name();
        let out_name = session.output(0).name();

        let ro = RunOptions::new();

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_tensor = Tensor::new(&[3, 2], input_data)?;

        // immutable version
        let output = session.run_raw(&ro, &[&in_name], &[input_tensor.value()], &[&out_name])?;

        let output_value = output.into_iter().next().unwrap();
        let output_tensor = output_value.as_tensor::<f32>().ok().expect("as_tensor");

        assert_eq!(
            &output_tensor[..],
            &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]
        );

        // mutable version
        let mut output_tensor = Tensor::<f32>::init(&[3, 1], 0.0)?;

        run!(session, &ro, "X": &input_tensor, "Y": &mut output_tensor)?;

        assert_eq!(
            &output_tensor[..],
            &[1. + 2. * 2., 3. + 2. * 4., 5. + 2. * 6.]
        );

        Ok(())
    }
}
