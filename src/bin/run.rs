use std::time::{Duration, Instant};

use onnxruntime::*;
use structopt::{clap, StructOpt};

#[structopt(
    name = "run",
    about = "Run a benchmark on an onnx model. Each worker runs the model in a loop in its own
    thead. Once done it will print the average time to run the model.",
    setting = clap::AppSettings::ColoredHelp
)]
#[derive(StructOpt)]
struct Opt {
    /// The path to the onnx files to benchmark
    onnx: Vec<String>,

    /// A comma separated list of symbolic_dimension=value. If a symbolic dimension is not
    /// specified, 1 will be used.
    #[structopt(long)]
    dims: Option<String>,

    /// The number of worker threads to spawn
    #[structopt(long, default_value = "1")]
    workers: usize,

    /// The number of runs each worker will
    #[structopt(long, default_value = "1")]
    runs: usize,
}

use std::collections::HashMap;

fn key_val_parse(str: &str) -> HashMap<String, usize> {
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
        map.insert(key.to_owned(), val);
    }
    map
}

/// Get the size of a tensor, substituting symbolic dimentions.
fn tensor_size(
    info: &TensorInfo,
    named_sizes: &mut HashMap<String, usize>,
) -> (OnnxTensorElementDataType, Vec<usize>) {
    let dims = info
        .symbolic_dims()
        .map(|d| match d {
            SymbolicDim::Symbolic(name) => {
                let name = name.to_str().unwrap();
                named_sizes.get(name).cloned().unwrap_or_else(|| {
                    eprintln!("name {} not specified, setting to 1", name);
                    named_sizes.insert(name.to_owned(), 1);
                    1
                })
            }
            SymbolicDim::Fixed(x) => x,
        })
        .collect();
    (info.elem_type(), dims)
}

fn tensor_mut(elem_type: OnnxTensorElementDataType, dims: &[usize]) -> Box<dyn AsMut<Val>> {
    use OnnxTensorElementDataType::*;
    match elem_type {
        Float => Box::new(Tensor::<f32>::init(dims, 0.0).unwrap()),
        Int64 => Box::new(Tensor::<i64>::init(dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

fn tensor_with_size(
    info: &TensorInfo,
    named_sizes: &mut HashMap<String, usize>,
) -> Box<dyn AsRef<Val> + Sync> {
    let (ty, dims) = tensor_size(info, named_sizes);
    use OnnxTensorElementDataType::*;
    match ty {
        Float => Box::new(Tensor::<f32>::init(&dims, 0.0).unwrap()),
        Int64 => Box::new(Tensor::<i64>::init(&dims, 0).unwrap()),
        t => panic!("Unsupported type {:?}", t),
    }
}

fn main() -> Result<()> {
    let env = Env::new(LoggingLevel::Fatal, "test")?;
    let opt = Opt::from_args();

    let so = SessionOptions::new()?;

    let mut map = if let Some(dims) = &opt.dims {
        key_val_parse(dims)
    } else {
        HashMap::new()
    };

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

        let mut input_names: Vec<OrtString> = vec![];
        let mut input_tensors: Vec<Box<dyn AsRef<Val> + Sync>> = vec![];

        for (i, input) in session.inputs().enumerate() {
            if let Some(tensor_info) = input.tensor_info() {
                input_names.push(input.name());
                input_tensors.push(tensor_with_size(&tensor_info, &mut map));
            } else {
                println!("input {}: {:?} {:?}", i, &*input.name(), input.onnx_type());
            }
        }

        let mut output_names: Vec<OrtString> = vec![];
        let mut output_sizes: Vec<(OnnxTensorElementDataType, Vec<usize>)> = vec![];

        for (i, output) in session.outputs().enumerate() {
            if let Some(tensor_info) = output.tensor_info() {
                output_names.push(output.name());
                output_sizes.push(tensor_size(&tensor_info, &mut map));
            } else {
                println!(
                    "output {}: {:?} {:?}",
                    i,
                    &*output.name(),
                    output.onnx_type()
                );
            }
        }

        crossbeam::scope(|s| {
            let mut workers = vec![];
            for i in 0..opt.workers {
                let i = std::sync::Arc::new(i);
                workers.push(s.spawn(|_| {
                    let i = i;
                    let ro = RunOptions::new();
                    // allocate output vectors
                    let mut output_tensors: Vec<_> = output_sizes
                        .iter()
                        .map(|(elem_type, size)| tensor_mut(*elem_type, size))
                        .collect();

                    let inputs = input_names
                        .iter()
                        .zip(input_tensors.iter())
                        .map(|(nm, val)| (nm.as_str(), val.as_ref().as_ref()));

                    let outputs = output_names
                        .iter()
                        .zip(output_tensors.iter_mut())
                        .map(|(nm, val)| (nm.as_str(), val.as_mut().as_mut()));

                    // warmup run
                    session.run(&ro, inputs, outputs).expect("run");

                    let mut times = vec![];
                    for _ in 0..opt.runs {
                        let before = Instant::now();
                        let inputs = input_names
                            .iter()
                            .zip(input_tensors.iter())
                            .map(|(nm, val)| (nm.as_str(), val.as_ref().as_ref()));

                        let outputs = output_names
                            .iter()
                            .zip(output_tensors.iter_mut())
                            .map(|(nm, val)| (nm.as_str(), val.as_mut().as_mut()));

                        session.run(&ro, inputs, outputs).expect("run");
                        times.push(before.elapsed());
                    }
                    let total: Duration = times.iter().sum();
                    let avg = total / (times.len() as u32);
                    eprintln!("worker {} avg time: {:.2} ms", i, avg.as_secs_f64() * 1e3);
                }));
            }
            workers.into_iter().for_each(|j| j.join().unwrap());
        })
        .unwrap();
    }

    Ok(())
}
