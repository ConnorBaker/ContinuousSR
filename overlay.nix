final: prev: {
  pythonPackagesExtensions = [
    (finalPythonPackages: _: {
      basicsr = finalPythonPackages.callPackage ./basicsr.nix { };
      gsplat = finalPythonPackages.callPackage ./gsplat.nix { };
    })
  ];
}
