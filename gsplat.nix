{
  buildPythonPackage,
  fetchFromGitHub,
  jaxtyping,
  lib,
  rich,
  setuptools,
  torch,
  typing-extensions,
}:
buildPythonPackage {
  pname = "gsplat";
  version = "0-unstable-2024-07-04";

  pyproject = true;

  inherit (torch) stdenv;

  src = fetchFromGitHub {
    owner = "XingtongGe";
    repo = "gsplat";
    rev = "bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d";
    hash = "sha256-J/qEzO2jcAf0EyE7l51Rgl9R5HiDOuxwYE6VQQF70sw=";
  };

  env = {
    FORCE_CUDA = 1;
    TORCH_CUDA_ARCH_LIST = "${lib.concatStringsSep ";" torch.cudaCapabilities}";
  };

  nativeBuildInputs = [
    torch.cudaPackages.cuda_nvcc
  ];

  build-system = [
    setuptools
  ];

  buildInputs = [
    torch.cxxdev
  ];

  dependencies = [
    jaxtyping
    rich
    torch
    typing-extensions
  ];

  # TODO: tests

  pythonImportsCheck = [ "gsplat" ];

  meta = {
    description = "Submodule for GaussianImage";
    homepage = "https://github.com/XingtongGe/gsplat";
    license = lib.licenses.asl20;
    broken = !torch.cudaSupport;
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
    maintainers = with lib.maintainers; [ ConnorBaker ];
  };
}
