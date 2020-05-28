use std::fmt;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

use crate::*;

/// This is what `CStr` is to `CString` for `Value`. The motivating use case for this is that
/// `&[&Val]` can be converted to `*const sys::Value` at zero cost.
pub struct Val {
    raw: sys::Value,
}

impl Deref for Value {
    type Target = Val;
    fn deref(&self) -> &Val {
        unsafe { &*(self.raw as *const sys::Value as *const Val) }
    }
}

impl DerefMut for Value {
    fn deref_mut(&mut self) -> &mut Val {
        unsafe { &mut *(self.raw as *mut Val) }
    }
}

impl Val {
    pub fn raw(&self) -> *mut sys::Value {
        &self.raw as *const sys::Value as *mut sys::Value
    }

    pub fn is_tensor(&self) -> bool {
        call!(@unsafe @int @expect IsTensor, self.raw()) == 1
    }

    pub fn tensor_data(&self) -> *mut c_void {
        call!(@unsafe @ptr @expect GetTensorMutableData, self.raw())
    }

    pub fn shape_and_type(&self) -> TensorTypeAndShapeInfo {
        let raw = call!(@unsafe @ptr @expect GetTensorTypeAndShape, self.raw());
        TensorTypeAndShapeInfo { raw }
    }
}

impl Value {
    /// Create a tensor from an allocator.
    pub fn alloc_tensor(
        alloc: Allocator,
        shape: &[i64],
        data_type: OnnxTensorElementDataType,
    ) -> Result<Value> {
        let raw = call!(@unsafe @ptr
            CreateTensorAsOrtValue,
            alloc.as_ptr(),
            shape.as_ptr(),
            shape.len() as u64,
            data_type
        )?;
        Ok(Value { raw })
    }

    pub fn as_tensor<T: OrtType>(self) -> std::result::Result<Tensor<T>, Self> {
        let shape_and_type = self.shape_and_type();
        if !self.is_tensor() || shape_and_type.elem_type() != T::onnx_type() {
            return Err(self);
        }
        Ok(Tensor {
            owned: None,
            val: self,
            shape: shape_and_type.dims(),
        })
    }
}

/// Just like `Val` for `Value`, we have `TensorInfo` for `TensorTypeAndShapeInfo`, an unowned
/// reference.
pub struct TensorInfo {
    raw: sys::TensorTypeAndShapeInfo,
}

impl Deref for TensorTypeAndShapeInfo {
    type Target = TensorInfo;
    fn deref(&self) -> &TensorInfo {
        unsafe { TensorInfo::from_raw(self.raw) }
    }
}

impl DerefMut for TensorTypeAndShapeInfo {
    fn deref_mut(&mut self) -> &mut TensorInfo {
        unsafe { &mut *(self.raw as *mut TensorInfo) }
    }
}

impl TensorInfo {
    pub unsafe fn from_raw<'a>(raw: *const sys::TensorTypeAndShapeInfo) -> &'a TensorInfo {
        &*(raw as *const TensorInfo)
    }

    pub fn raw(&self) -> *mut sys::TensorTypeAndShapeInfo {
        &self.raw as *const sys::TensorTypeAndShapeInfo as *mut _
    }

    /// Get the number of dimentions in this tensor.
    pub fn num_dims(&self) -> usize {
        call!(@unsafe @int @expect GetDimensionsCount, self.raw()) as usize
    }

    pub fn dims(&self) -> Vec<i64> {
        let mut dims = vec![0; self.num_dims() as usize];
        call!(@unsafe @expect
            GetDimensions,
            self.raw(),
            dims.as_mut_ptr(),
            dims.len() as u64
        );
        dims
    }

    pub unsafe fn set_dims(&mut self, dims: &[i64]) {
        call!(@expect SetDimensions, self.raw(), dims.as_ptr(), dims.len() as u64)
    }

    /// Return the number of elements specified by the tensor shape. Return a negative value if
    /// unknown (i.e., any dimension is negative.)
    ///
    /// ```text
    /// [] -> 1
    /// [1,3,4] -> 12
    /// [2,0,4] -> 0
    /// [-1,3,4] -> -1
    /// ```
    pub fn elem_count(&self) -> isize {
        call!(@unsafe @int @expect GetTensorShapeElementCount, self.raw()) as isize
    }

    /// Get the data type for the elements of this tensor.
    pub fn elem_type(&self) -> OnnxTensorElementDataType {
        call!(@unsafe @arg OnnxTensorElementDataType::Undefined; @expect
              GetTensorElementType, self.raw())
    }

    /// Iterate over the names of the symbolic dimentions.
    pub fn symbolic_dims(&self) -> impl ExactSizeIterator<Item = SymbolicDim<&CStr>> {
        let fixeds = self.dims();
        let mut symbolics = vec![ptr::null(); self.num_dims() as usize];
        call!(@unsafe @expect
            GetSymbolicDimensions,
            self.raw(),
            symbolics.as_mut_ptr(),
            symbolics.len() as u64
        );
        fixeds.into_iter().zip(symbolics).map(|(fixed, ptr)| {
            if fixed >= 0 {
                SymbolicDim::Fixed(fixed as usize)
            } else {
                SymbolicDim::Symbolic(unsafe { CStr::from_ptr(ptr) })
            }
        })
    }
}

pub enum SymbolicDim<T> {
    Symbolic(T),
    Fixed(usize),
}

impl<T> SymbolicDim<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> SymbolicDim<U> {
        use SymbolicDim::*;
        match self {
            Symbolic(a) => Symbolic(f(a)),
            Fixed(u) => Fixed(u),
        }
    }
}


impl<T: fmt::Debug> fmt::Debug for SymbolicDim<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicDim::Symbolic(t) => write!(fmt, "{:?}", t),
            SymbolicDim::Fixed(u) => write!(fmt, "{}", u),
        }
    }
}

pub struct Tensor<T> {
    /// If this is none then ort owns the data.
    owned: Option<Vec<T>>,
    val: Value,
    shape: Vec<i64>,
}

pub unsafe trait OrtType: Sized {
    fn onnx_type() -> OnnxTensorElementDataType;
}

macro_rules! ort_data_type {
    ($t:ident, $ty:ty) => {
        unsafe impl OrtType for $ty {
            fn onnx_type() -> OnnxTensorElementDataType {
                use OnnxTensorElementDataType::*;
                $t
            }
        }
    };
}

// missing: String, Complex64, Complex128, BFloat16, Undefined
ort_data_type!(Float, f32);
ort_data_type!(Double, f64);
ort_data_type!(Uint8, u8);
ort_data_type!(Int8, i8);
ort_data_type!(Uint16, u16);
ort_data_type!(Int16, i16);
ort_data_type!(Uint32, u32);
ort_data_type!(Int32, i32);
ort_data_type!(Uint64, u64);
ort_data_type!(Int64, i64);
ort_data_type!(Bool, bool);

impl<T: OrtType> Tensor<T> {
    /// Create a new tensor with the given shape and data.
    pub fn new(shape: Vec<i64>, mut vec: Vec<T>) -> Result<Tensor<T>> {
        let raw = call!(@unsafe @ptr
            CreateTensorWithDataAsOrtValue,
            CPU_ARENA.raw,
            vec.as_mut_ptr() as *mut _,
            (vec.len() * std::mem::size_of::<T>()) as u64,
            shape.as_ptr(),
            shape.len() as u64,
            T::onnx_type()
        )?;
        Ok(Tensor {
            owned: Some(vec),
            val: Value { raw },
            shape,
        })
    }

    pub fn init(shape: Vec<i64>, value: T) -> Result<Tensor<T>>
    where
        T: Copy,
    {
        let len = shape.iter().product::<i64>();
        Self::new(shape, vec![value; len as usize])
    }

    pub fn dims(&self) -> &[i64] {
        &self.shape
    }

    pub fn value(&self) -> &Value {
        &self.val
    }

    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.val
    }

    // must be owned or will panic, don't give it negative dims
    pub fn resize(&mut self, dims: Vec<i64>)
    where
        T: Clone + Default,
    {
        let len = dims.iter().product::<i64>();
        let owned = self.owned.as_mut().expect("Tensor::resize not owned");
        owned.resize(len as usize, T::default());
        unsafe {
            self.value_mut().shape_and_type().set_dims(&dims);
        }
    }
}

impl<T> std::borrow::Borrow<[T]> for Tensor<T> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

impl<T> std::borrow::BorrowMut<[T]> for Tensor<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<T> std::ops::Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        if let Some(v) = &self.owned {
            &v
        } else {
            let len: i64 = self.shape.iter().cloned().product();
            // don't return anything when the shape isn't known
            if len <= 0 {
                &[]
            } else {
                unsafe {
                    std::slice::from_raw_parts(self.val.tensor_data() as *const _, len as usize)
                }
            }
        }
    }
}

impl<T> std::ops::DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if let Some(v) = &mut self.owned {
            v
        } else {
            let len: i64 = self.shape.iter().cloned().product();
            // don't return anything when the shape isn't known
            if len <= 0 {
                &mut []
            } else {
                unsafe {
                    std::slice::from_raw_parts_mut(self.val.tensor_data() as *mut _, len as usize)
                }
            }
        }
    }
}

impl<T> std::convert::AsRef<Val> for Tensor<T> {
    fn as_ref(&self) -> &Val {
        &self.val
    }
}

impl<T> std::convert::AsMut<Val> for Tensor<T> {
    fn as_mut(&mut self) -> &mut Val {
        &mut self.val
    }
}
