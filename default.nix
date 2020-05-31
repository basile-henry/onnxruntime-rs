{ nixpkgs ? import ./nix/nixpkgs.nix }:

{
  # Just onnxruntime-rs with no onnxruntime dependency
  # Tests are not run
  onnxruntime-rs = import ./onnxruntime-rs.nix {
    inherit nixpkgs;
  };

  onnxruntime-rs-tested = import ./onnxruntime-rs.nix {
    inherit nixpkgs;
    inherit (nixpkgs) onnxruntime;
  };
}
