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
          pkgs.python313
          pkgs.uv
          pkgs.ruff
        ];

        # numpy 依存ライブラリへの PATH
        # 他にも uv add <パッケージ名> をした後に実行してランタイムエラーになる場合,
        # ここにライブラリを書けば解決するかもしれない
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          libgcc.lib
          libz
        ]);

        shellHook = ''
          uv sync
          source .venv/bin/activate
        '';
      };
    });
}
