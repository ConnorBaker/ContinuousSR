{
  addict,
  buildPythonPackage,
  cython,
  distutils,
  fetchFromGitHub,
  lib,
  lmdb,
  opencv4,
  scikit-image,
  setuptools,
  torch,
  torchvision,
  tqdm,
  yapf,
}:
buildPythonPackage {
  pname = "basicsr";
  version = "1.4.2-unstable-2024-05-17";

  pyproject = true;

  inherit (torch) stdenv;

  src = fetchFromGitHub {
    owner = "XPixelGroup";
    repo = "BasicSR";
    rev = "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a";
    hash = "sha256-05rysJfjFfHXPZZvEdD58GP7pk+EZZX6P0h2NrrENvw=";
  };

  # TODO: These seem to be ignored; I don't see any compilation of CUDA code.
  env = {
    BASICSR_EXT = 1;
    FORCE_CUDA = 1;
    TORCH_CUDA_ARCH_LIST = "${lib.concatStringsSep ";" torch.cudaCapabilities}";
  };

  build-system = [
    cython
    setuptools
  ];

  # NOTE: Their approach to getting the version dynamically is broken (and gross).
  # We need to update this whenever the version is updated.
  prePatch = ''
    nixLog "patching $PWD/setup.py to fix version information"
    substituteInPlace "$PWD/setup.py" \
      --replace-fail \
        'version=get_version(),' \
        'version="1.4.2",'
  '';

  pythonRemoveDeps = [
    "future" # TODO: Not supported for Python3.13
    "opencv-python" # opencv4
    "tb-nightly" # TODO
  ];

  dependencies = [
    addict
    distutils
    lmdb
    opencv4
    scikit-image
    torch
    torchvision
    tqdm
    yapf
  ];

  # TODO: Tests

  pythonImportsCheck = [ "basicsr" ];

  meta = {
    description = "Open-source image and video restoration toolbox based on PyTorch";
    homepage = "https://github.com/XPixelGroup/BasicSR";
    license = lib.licenses.asl20;
    broken = !torch.cudaSupport;
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
    maintainers = with lib.maintainers; [ ConnorBaker ];
  };
}
