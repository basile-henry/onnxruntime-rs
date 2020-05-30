lazy_static::lazy_static! {
    pub(crate) static ref API: &'static crate::sys::Api = unsafe {
        let api_base = crate::sys::GetApiBase().as_ref().unwrap();
        let get_api = api_base.GetApi.unwrap();
        get_api(crate::sys::ORT_API_VERSION).as_ref().unwrap()
    };
}

#[macro_use]
macro_rules! call {
    // the raw call
    (@raw $name:ident, $($arg:expr),*) => {
        (crate::api::API.$name.expect(concat!("ORT api: \"", stringify!($name), "\" unavailable", )))($($arg),*)
    };

    // the checked call that returns Ok(res) on success
    ($name:ident, $($arg:expr),* => $res:expr) => {{
        let status = call!(@raw $name, $($arg),*);
        match crate::Status::from_raw(status) {
            Some(status) => std::result::Result::Err(crate::Error::OrtError(status)),
            None => std::result::Result::Ok($res),
        }
    }};

    // leading expect means expect
    (@unsafe $($rest:tt)*) => {{
        unsafe { call!($($rest)*) }
    }};

    // leading expect means expect
    (@expect $name:ident, $($rest:tt)*) => {{
        call!($name, $($rest)*).expect(stringify!($name))
    }};

    // checked call without a return
    ($name:ident, $($arg:expr),*) => { call!($name, $($arg),* => ()) };

    // types that use the last argument for the call by mut ref return type

    // the type to initialise with the @type syntax
    (@int $($rest:tt)*) => { call!(@arg 0; $($rest)*) };
    (@ptr $($rest:tt)*) => { call!(@arg ::std::ptr::null_mut(); $($rest)*) };

    // no arguments
    (@arg $var:expr; $(@$expect:ident)* $name:ident) => {{
        let mut var = $var;
        call!($(@$expect)* $name, &mut var => var)
    }};
    // multiple arguments
    (@arg $var:expr; $(@$expect:ident)* $name:ident, $($arg:expr),*) => {{
        let mut var = $var;
        call!($(@$expect)* $name, $($arg),*, &mut var => var)
    }};
}
