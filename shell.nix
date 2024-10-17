let
  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/24bb1b20a9a57175965c0a9fb9533e00e370c88b.tar.gz") {config.allowUnfree = true; };
  python = pkgs.python312;

  docstring-parser = python.pkgs.buildPythonPackage rec {
    pname = "docstring_parser";
    version = "0.16";
    format = "pyproject";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "U4vqvQrx4tsBRra9PKpSbDWjTWGvn9KIfzqKJ6c5qm4=";
    };

    propagatedBuildInputs = [
      pkgs.python312Packages.poetry-core
    ];
  };

  arguably = python.pkgs.buildPythonPackage rec {
    pname = "arguably";
    version = "1.3.0";
    format = "pyproject";
    doCheck = false;
    src = pkgs.fetchFromGitHub {
      owner = "treykeown";
      repo = "arguably";
      rev = "v1.3.0";
      sha256 = "CumvE8o1mJQ/uwZ6ZikqOKOygAllTdw5jyt09DNdMUE=";
    };

    # Add poetry-core as a build input
    propagatedBuildInputs = [
      pkgs.python312Packages.poetry-core
      docstring-parser
    ];
  };

  python-with-packages = python.withPackages (ps: with ps; [
    torch
    torchaudio
    torch-audiomentations
    librosa
    jiwer
    datasets
    transformers
    evaluate
    accelerate
    pip
    pytorch-lightning
    arguably
  ]);
in pkgs.mkShell {
  packages = [
    python-with-packages
  ];

  shellHook = ''
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export CUDA_PATH=${pkgs.cudatoolkit}
  '';
}