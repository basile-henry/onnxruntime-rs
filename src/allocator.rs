use std::ffi::c_void;

use crate::sys;

pub struct Allocator {
    raw: *mut sys::Allocator,
}

impl Default for Allocator {
    fn default() -> Self {
        let raw = call!(@unsafe @ptr @expect GetAllocatorWithDefaultOptions);
        Allocator { raw }
    }
}

impl Allocator {
    pub fn as_ptr(&self) -> *mut sys::Allocator {
        self.raw
    }

    pub unsafe fn free(&self, ptr: *mut c_void) {
        call!(@expect AllocatorFree, self.raw, ptr);
    }
}
