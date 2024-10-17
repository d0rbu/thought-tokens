let
  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/24bb1b20a9a57175965c0a9fb9533e00e370c88b.tar.gz") {config.allowUnfree = true; };
  python = pkgs.python312;

  arguably = python.pkgs.buildPythonPackage rec {
    pname = "arguably";
    version = "1.3.0";
    format = "wheel";
    doCheck = false;
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "9261e49d0281600e9eac3fb2e31d2022dc0d002b6370461d787b20690eb2a98d";
    };
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