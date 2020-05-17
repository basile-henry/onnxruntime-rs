use std::ffi::c_void;
use std::ptr;

use crate::sys;

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
    pub fn as_ptr(&self) -> *mut sys::Allocator {
        self.raw
    }

    pub unsafe fn free(&self, ptr: *mut c_void) {
        checked_call!(AllocatorFree, self.raw, ptr).expect("AllocatorFree");
    }
}
