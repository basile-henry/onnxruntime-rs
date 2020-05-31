use onnxruntime::*;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    /// The path to the gru onnx file
    // #[structopt(long)]
    onnx: Vec<String>,
    // #[structopt(long)]
    // ort_profile: Option<String>,

    // #[structopt(long, default_value="1")]
    // batch_size: i64,

    // #[structopt(long, default_value="1")]
    // runs: usize,

    // #[structopt(long, default_value="1")]
    // workers: usize,
}

fn main() -> Result<()> {
    let env = Env::new(LoggingLevel::Fatal, "test")?;
    let opt = Opt::from_args();

    let so = SessionOptions::new()?;

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

        for (i, input) in session.inputs().enumerate() {
            if let Some(tensor_info) = input.tensor_info() {
                let dims = tensor_info.symbolic_dims().collect::<Vec<_>>();
                println!(
                    "input {}: {:?} {:?} {:?}",
                    i,
                    &*input.name(),
                    dims,
                    tensor_info.elem_type()
                )
            } else {
                println!("input {}: {:?} {:?}", i, &*input.name(), input.onnx_type());
            }
        }
        for (i, output) in session.outputs().enumerate() {
            if let Some(tensor_info) = output.tensor_info() {
                let dims = tensor_info.symbolic_dims().collect::<Vec<_>>();
                println!(
                    "output {}: {:?} {:?} {:?}",
                    i,
                    &*output.name(),
                    dims,
                    tensor_info.elem_type()
                )
            } else {
                println!(
                    "input {}: {:?} {:?}",
                    i,
                    &*output.name(),
                    output.onnx_type()
                );
            }
        }
        for (i, output) in session.overridable_initializers().enumerate() {
            println!("init {}: {:?}", i, &*output.name())
        }
        println!();
    }

    Ok(())
}
