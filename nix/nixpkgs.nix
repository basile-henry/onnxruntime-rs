let sources = import ./sources.nix;

    moz-overlay = import sources.nixpkgs-mozilla;

    overlay = _: pkgs: {
      naersk = pkgs.callPackage sources.naersk {};
      niv = pkgs.callPackage sources.niv {};
    };
in
import sources.nixpkgs {
  overlays = [ moz-overlay overlay ];
}
