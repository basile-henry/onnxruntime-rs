use onnxruntime::*;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    /// The path to the gru onnx file
    // #[structopt(long)]
    onnx: Vec<String>,

    #[structopt(long, default_value = "")]
    dims: String,
    // #[structopt(long, default_value="1")]
    // workers: usize,
}

use std::collections::HashMap;

fn key_val_parse(str: &str) -> HashMap<&str, usize> {
    let mut map = HashMap::new();
    if str.is_empty() {
        return map;
    }
    for key_val in str.split(',') {
        let mut iter = key_val.split('=');
        let key = iter.next().expect("no =");
        let val = iter
            .next()
            .expect("nothing after =")
            .parse()
            .expect("parse error");
        assert!(iter.next().is_none(), "more than 1 =");
        map.insert(key, val);
    }
    map
}

fn tensor_with_size(info: &TensorInfo, named_sizes: &HashMap<&str, usize>) -> Box<dyn AsRef<Val>> {
    use OnnxTensorElementDataType::*;
    let dims = info
        .symbolic_dims()
        .map(|d| match d {
            SymbolicDim::Symbolic(name) => {
                let name = name.to_str().unwrap();
                *named_sizes.get(&name).unwrap_or_else(|| {
                    eprintln!("name {} not specified, setting to 1", name);
                    &1
                })
            }
            SymbolicDim::Fixed(x) => x,
        })
        .collect::<Vec<usize>>();
    match info.elem_type() {
        Float => Box::new(Tensor::<f32>::init(&dims, 0.0).unwrap()),
        Int64 => Box::new(Tensor::<i64>::init(&dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

fn tensor_with_size_mut(
    info: &TensorInfo,
    named_sizes: &HashMap<&str, usize>,
) -> Box<dyn AsMut<Val>> {
    use OnnxTensorElementDataType::*;
    let dims = info
        .symbolic_dims()
        .map(|d| match d {
            SymbolicDim::Symbolic(name) => {
                let name = name.to_str().unwrap();
                *named_sizes.get(&name).unwrap_or_else(|| {
                    eprintln!("name {} not specified, setting to 1", name);
                    &1
                })
            }
            SymbolicDim::Fixed(x) => x,
        })
        .collect::<Vec<usize>>();
    match info.elem_type() {
        Float => Box::new(Tensor::<f32>::init(&dims, 0.0).unwrap()),
        Int64 => Box::new(Tensor::<i64>::init(&dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

use std::ffi::{CStr, CString};

fn main() -> Result<()> {
    let env = Env::new(LoggingLevel::Fatal, "test")?;
    let opt = Opt::from_args();

    let so = SessionOptions::new()?;

    let map = key_val_parse(&opt.dims);

    for path in &opt.onnx {
        println!("model {:?}", path);
        let session = match Session::new(&env, path, &so) {
            Ok(sess) => sess,
            Err(err) => {
                eprintln!("error: {}\n", err);
                continue;
            }
        };

        let metadata = session.metadata();
        eprintln!("name: {}", metadata.producer_name());
        eprintln!("graph_name: {}", metadata.graph_name());
        eprintln!("domain: {}", metadata.domain());
        eprintln!("description: {}", metadata.description());

        let mut input_names: Vec<CString> = vec![];
        let mut input_tensors: Vec<Box<dyn AsRef<Val>>> = vec![];
        let mut output_names: Vec<CString> = vec![];
        let mut output_tensors: Vec<Box<dyn AsMut<Val>>> = vec![];

        for (i, input) in session.inputs().enumerate() {
            if let Some(tensor_info) = input.tensor_info() {
                input_names.push(input.name().to_owned());
                input_tensors.push(tensor_with_size(&tensor_info, &map));
            } else {
                println!("input {}: {:?} {:?}", i, &*input.name(), input.onnx_type());
            }
        }
        for (i, output) in session.outputs().enumerate() {
            if let Some(tensor_info) = output.tensor_info() {
                output_names.push(output.name().to_owned());
                output_tensors.push(tensor_with_size_mut(&tensor_info, &map));
            } else {
                println!(
                    "output {}: {:?} {:?}",
                    i,
                    &*output.name(),
                    output.onnx_type()
                );
            }
        }

        let so = RunOptions::new();

        let in_names: Vec<&CStr> = input_names.iter().map(|x| x.as_c_str()).collect();
        let in_vals: Vec<&Val> = input_tensors.iter().map(|x| x.as_ref().as_ref()).collect();
        let out_names: Vec<&CStr> = output_names.iter().map(|x| x.as_c_str()).collect();
        let mut out_vals: Vec<&mut Val> = output_tensors
            .iter_mut()
            .map(|x| x.as_mut().as_mut())
            .collect();

        session
            .run_mut(&so, &in_names, &in_vals[..], &out_names, &mut out_vals[..])
            .expect("run");

        // pub fn run_mut(
        //     &self,
        //     options: &RunOptions,
        //     input_names: &[&CStr],
        //     inputs: &[&Val],
        //     output_names: &[&CStr],
        //     outputs: &mut [&mut Val],

        for (i, output) in session.overridable_initializers().enumerate() {
            println!("init {}: {:?}", i, &*output.name())
        }
        println!();
    }

    Ok(())
}
