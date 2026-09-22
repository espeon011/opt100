{
  description = "python with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem
    (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
    in {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = [];
        BuildInputs = [];
        packages = [
          pkgs.python314
          pkgs.uv
          pkgs.pyrefly
          pkgs.ty
          pkgs.ruff
          pkgs.marimo
          pkgs.unzip
        ];

        # uv で追加したパッケージがランタイムエラーを引き起こす場合,
        # ここにライブラリを書けば解決するかもしれない
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          libgcc.lib
          zlib
        ]);

        # 内部で pypi 側の marimo を使うようにする
        shellHook = ''
          unset PYTHONPATH
        '';
      };
    });
}
