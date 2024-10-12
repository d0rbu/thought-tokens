let
  # Pin to a specific nixpkgs commit for reproducibility.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/24bb1b20a9a57175965c0a9fb9533e00e370c88b.tar.gz") {config.allowUnfree = true; };
in pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.python311
    pkgs.python311Packages.torch-bin
    pkgs.python311Packages.torchaudio-bin
    pkgs.python311Packages.torch-audiomentations
    pkgs.python311Packages.librosa
    pkgs.python311Packages.jiwer
    pkgs.python311Packages.datasets
    pkgs.python311Packages.transformers
    pkgs.python311Packages.evaluate
    pkgs.python311Packages.accelerate
    pkgs.python311Packages.pip
    
  ];

  shellHook = ''
    echo "You are now using a NIX environment"
    export CUDA_PATH=${pkgs.cudatoolkit}
    echo $CUDA_PATH
  '';
}