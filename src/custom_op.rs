use std::ffi::c_void;

use super::*;

impl CustomOpDomain {
    pub fn new(domain: &str) -> CustomOpDomain {
        let domain = CString::new(domain).unwrap();
        let raw = call!(@unsafe @ptr @expect CreateCustomOpDomain, domain.as_ptr());
        CustomOpDomain { raw }
    }

    pub fn add<Op>(&mut self, op: &mut CustomOpWithContext<Op>) {
        call!(@unsafe @expect CustomOpDomain_Add, self.raw, op.as_ptr());
    }
}

pub trait Kernel {
    /// The compute function for a custom op kernel. The KernelContext is used to obtain the inputs
    /// and create the outputs.
    fn compute(&mut self, context: &mut KernelContext);
}

pub trait CustomOperation {
    type CustomKernel: Kernel;
    fn create_kernel(&mut self, info: &KernelInfo) -> Self::CustomKernel;
    fn name(&self) -> &str;
    fn execution_provider_type(&self) -> Option<&str>;
    fn input_type(&self, index: usize) -> OnnxTensorElementDataType;
    fn input_type_count(&self) -> usize;
    fn output_type(&self, index: usize) -> OnnxTensorElementDataType;
    fn output_type_count(&self) -> usize;
}

impl KernelInfo {
    pub fn attribute_float(&self, name: &str) -> Result<f32> {
        let name = CString::new(name).expect("KernelInfo::attribute_float");
        call!(@unsafe @arg 0.0; KernelInfoGetAttribute_float, self, name.as_ptr())
    }
    pub fn attribute_i64(&self, name: &str) -> Result<i64> {
        let name = CString::new(name).expect("KernelInfo::attribute_i64");
        call!(@unsafe @int KernelInfoGetAttribute_int64, self, name.as_ptr())
    }
}

impl KernelContext {
    pub fn input_count(&self) -> usize {
        call!(@unsafe @int @expect KernelContext_GetInputCount, self) as usize
    }
    pub fn output_count(&self) -> usize {
        call!(@unsafe @int @expect KernelContext_GetOutputCount, self) as usize
    }
    pub fn get_input(&self, index: usize) -> &Val {
        let raw =
            call!(@unsafe @arg ptr::null(); @expect KernelContext_GetInput, self, index as u64);
        unsafe { &*(raw as *const Val) }
    }
    pub fn get_output(&mut self, index: usize, dims: &[usize]) -> &mut Val {
        let dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let raw = call!(@unsafe @ptr @expect KernelContext_GetOutput,
                        self,
                        index as u64,
                        dims.as_ptr(),
                        dims.len() as u64);
        unsafe { &mut *(raw as *mut Val) }
    }
}

/// A custom op with the context following it. This is done in a C struct because the internal
/// functions of `CustomOp` provide the pointer of the `CustomOp`. Adding a offset to this we can
/// get back the context.
#[repr(C)]
pub struct CustomOpWithContext<Ctx> {
    ort_custom_op: CustomOp,
    name: CString,
    execution_provider_type: Option<CString>,
    ctx: Ctx,
}

impl<Op> CustomOpWithContext<Op> {
    pub fn as_ptr(&mut self) -> *mut CustomOp {
        &mut self.ort_custom_op
    }
}

#[allow(non_snake_case)]
pub fn create_custom_op<Op: CustomOperation>(ctx: Op) -> CustomOpWithContext<Op> {
    unsafe extern "C" fn CreateKernel<Op: CustomOperation>(
        op: *mut CustomOp,
        api: *const sys::Api,
        info: *const KernelInfo,
    ) -> *mut c_void {
        let _ = api;
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        let info = info.as_ref().unwrap();
        let kernel = Box::new(op_with_ctx.ctx.create_kernel(info));
        Box::into_raw(kernel) as _
    }

    unsafe extern "C" fn GetName<Op: CustomOperation>(op: *mut CustomOp) -> *const c_char {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx.name.as_ptr()
    }

    unsafe extern "C" fn GetExecutionProviderType<Op: CustomOperation>(
        op: *mut CustomOp,
    ) -> *const c_char {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx
            .execution_provider_type
            .as_ref()
            .map_or(ptr::null(), |cstr| cstr.as_ptr())
    }

    unsafe extern "C" fn GetInputType<Op: CustomOperation>(
        op: *mut CustomOp,
        index: u64,
    ) -> OnnxTensorElementDataType {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx.ctx.input_type(index as usize)
    }

    unsafe extern "C" fn GetInputTypeCount<Op: CustomOperation>(op: *mut CustomOp) -> u64 {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx.ctx.input_type_count() as u64
    }

    unsafe extern "C" fn GetOutputType<Op: CustomOperation>(
        op: *mut CustomOp,
        index: u64,
    ) -> OnnxTensorElementDataType {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx.ctx.output_type(index as usize)
    }

    unsafe extern "C" fn GetOutputTypeCount<Op: CustomOperation>(op: *mut CustomOp) -> u64 {
        let op_with_ctx = (op as *mut CustomOpWithContext<Op>).as_mut().unwrap();
        op_with_ctx.ctx.output_type_count() as u64
    }

    unsafe extern "C" fn KernelCompute<Op: CustomOperation>(
        kernel: *mut c_void,
        context: *mut KernelContext,
    ) {
        let kernel = (kernel as *mut Op::CustomKernel).as_mut().unwrap();
        kernel.compute(context.as_mut().unwrap())
    }

    unsafe extern "C" fn KernelDestroy<Op: CustomOperation>(kernel: *mut c_void) {
        drop(Box::from_raw(kernel as *mut Op::CustomKernel));
    }

    let ort_custom_op = CustomOp {
        version: sys::ORT_API_VERSION,
        CreateKernel: Some(CreateKernel::<Op>),
        GetName: Some(GetName::<Op>),
        GetExecutionProviderType: Some(GetExecutionProviderType::<Op>),
        GetInputType: Some(GetInputType::<Op>),
        GetInputTypeCount: Some(GetInputTypeCount::<Op>),
        GetOutputType: Some(GetOutputType::<Op>),
        GetOutputTypeCount: Some(GetOutputTypeCount::<Op>),
        KernelCompute: Some(KernelCompute::<Op>),
        KernelDestroy: Some(KernelDestroy::<Op>),
    };

    CustomOpWithContext {
        ort_custom_op,
        name: CString::new(ctx.name()).unwrap(),
        execution_provider_type: ctx
            .execution_provider_type()
            .map(|str| CString::new(str).unwrap()),
        ctx,
    }
}
