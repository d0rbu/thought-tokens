let
  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/24bb1b20a9a57175965c0a9fb9533e00e370c88b.tar.gz") {config.allowUnfree = true;};
  python = pkgs.python311;
  pythonPackages = pkgs.python311Packages;

  docstring-parser = python.pkgs.buildPythonPackage rec {
    pname = "docstring_parser";
    version = "0.16";
    format = "pyproject";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "U4vqvQrx4tsBRra9PKpSbDWjTWGvn9KIfzqKJ6c5qm4=";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
    ];
  };

  arguably = python.pkgs.buildPythonPackage rec {
    pname = "arguably";
    version = "1.3.0";
    format = "pyproject";
    doCheck = false;
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "9261e49d0281600e9eac3fb2e31d2022dc0d002b6370461d787b20690eb2a98d";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
      docstring-parser
    ];
  };

  huggingface-hub = python.pkgs.buildPythonPackage rec {
    pname = "huggingface-hub[cli]";
    version = "0.25.2";
    format = "pyproject";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "a1014ea111a5f40ccd23f7f7ba8ac46e20fa3b658ced1f86a00c75c06ec6423c";
    };

    propagatedBuildInputs = [
      pythonPackages.poetry-core
    ];
  };

  python-with-packages = python.withPackages (ps: with ps;
    let
      accelerate-bin = accelerate.override { torch = torch-bin; };
      lightning-bin = pytorch-lightning.override { torch = torch-bin; };
    in [
      torch-bin
      torchvision-bin
      datasets
      transformers
      evaluate
      accelerate-bin
      pip
      lightning-bin
      arguably
      loguru
      tensorboard
      tensorboardx
      wandb
    ]);
in pkgs.mkShell {
  packages = [
    python-with-packages
  ];

  shellHook = ''
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export CUDA_PATH=${pkgs.cudatoolkit}

    source .env
    huggingface-cli login --token $HF_TOKEN
  '';
}