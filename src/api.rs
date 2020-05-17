lazy_static::lazy_static! {
    pub(crate) static ref API: &'static crate::sys::Api = unsafe {
        let api_base = crate::sys::GetApiBase().as_ref().unwrap();
        let get_api = api_base.GetApi.unwrap();
        get_api(crate::sys::ORT_API_VERSION).as_ref().unwrap()
    };
}

#[macro_use]
macro_rules! call {
    ($name:ident, $($arg:expr),*) => {
        (crate::api::API.$name.expect(concat!("ORT api: \"", stringify!($name), "\" unavailable", )))($($arg),*)
    }
}

#[macro_use]
macro_rules! checked_call {
    ($name:ident, $($arg:expr),*) => {{
        let status = call!($name, $($arg),*);
        match crate::Status::new(status) {
            Some(status) => std::result::Result::Err(crate::Error::OrtError(status)),
            None => std::result::Result::Ok(()),
        }
    }}
}
