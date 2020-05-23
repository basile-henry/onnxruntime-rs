use crate::*;
use std::ops::{Deref, DerefMut};

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
        let mut out = 0;
        unsafe {
            checked_call!(IsTensor, self.raw(), &mut out).expect("is_tensor");
        }
        out == 1
    }

    pub fn tensor_data(&self) -> *mut c_void {
        let mut data = ptr::null_mut();
        unsafe {
            checked_call!(GetTensorMutableData, self.raw(), &mut data)
                .expect("GetTensorMutableData");
        }
        data
    }

    pub fn shape_and_type(&self) -> TensorTypeAndShapeInfo {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(GetTensorTypeAndShape, self.raw(), &mut raw)
                .expect("TensorTypeAndShapeInfo");
        }
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
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(
                CreateTensorAsOrtValue,
                alloc.as_ptr(),
                shape.as_ptr(),
                shape.len() as u64,
                data_type,
                &mut raw
            )?;
        }
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

impl TensorTypeAndShapeInfo {
    pub fn dims(&self) -> Vec<i64> {
        let mut num_dims = 0;
        unsafe {
            checked_call!(GetDimensionsCount, self.raw, &mut num_dims)
                .expect("TensorTypeAndShapeInfo");
        }
        let mut dims = vec![0; num_dims as usize];
        unsafe {
            checked_call!(
                GetDimensions,
                self.raw,
                dims.as_mut_ptr(),
                dims.len() as u64
            )
            .expect("TensorTypeAndShapeInfo");
        }
        dims
    }

    pub unsafe fn set_dims(&mut self, dims: &[i64]) {
        checked_call!(SetDimensions, self.raw, dims.as_ptr(), dims.len() as u64)
            .expect("SetDimensions");
    }

    /// Return the number of elements specified by the tensor shape. Return a negative value if
    /// unknown (i.e., any dimension is negative.)
    ///
    /// [] -> 1
    /// [1,3,4] -> 12
    /// [2,0,4] -> 0
    /// [-1,3,4] -> -1
    pub fn elem_count(&self) -> isize {
        let mut count = 0;
        unsafe {
            checked_call!(GetTensorShapeElementCount, self.raw, &mut count).expect("SetDimensions");
        }
        // XXX check this, feels like the c_api signature is wrong, it's size_t even though it can
        // return negative numbers
        count as isize
    }

    pub fn elem_type(&self) -> OnnxTensorElementDataType {
        let mut info = OnnxTensorElementDataType::Undefined;
        unsafe {
            checked_call!(GetTensorElementType, self.raw, &mut info).expect("SetDimensions");
        }
        info
    }

    // no documentation for this?
    // pub fn symbolic_dims(&mut self) -> impl Iterator<Item=&str> {
    //     let mut dims = vec![0; num_dims as usize];
    //     unsafe {
    //         checked_call!(
    //             GetSymbolicDimensions,
    //             self.raw,
    //             dims.as_ptr(),
    //             dims.len() as u64
    //         ).expect("GetSymbolicDimensions");
    //     }
    // }
}

use std::ffi::c_void;

pub struct Tensor<T> {
    /// If this is none then ort owns the data.
    owned: Option<Vec<T>>,
    val: Value,
    shape: Vec<i64>,
}

pub trait OrtType: Sized {
    fn onnx_type() -> OnnxTensorElementDataType;
}

macro_rules! ort_data_type {
    ($t:ident, $ty:ty) => {
        impl OrtType for $ty {
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
    pub fn new(mem_info: MemoryInfo, shape: Vec<i64>, mut vec: Vec<T>) -> Result<Tensor<T>> {
        let mut raw = ptr::null_mut();
        unsafe {
            checked_call!(
                CreateTensorWithDataAsOrtValue,
                mem_info.raw,
                vec.as_mut_ptr() as *mut _,
                (vec.len() * std::mem::size_of::<T>()) as u64,
                shape.as_ptr(),
                shape.len() as u64,
                T::onnx_type(),
                &mut raw
            )?;
        }
        Ok(Tensor {
            owned: Some(vec),
            val: Value { raw },
            shape,
        })
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

#[macro_export]
/// A macro which expands a list of values or tensors into a vec of value
/// references. This is useful because you cannot have a vector of tensors of
/// different types.
macro_rules! values {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x.as_ref::<Value>());
            )*
            temp_vec
        }
    };
}
