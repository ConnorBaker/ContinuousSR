{
  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };

    # Use my branch with reworked CUDA packaging
    nixpkgs.url = "github:NixOS/nixpkgs/pull/437723/head";

    git-hooks-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/git-hooks.nix";
    };

    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  outputs =
    inputs:
    let
      inherit (inputs.flake-parts.lib) mkFlake;
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
    in
    mkFlake { inherit inputs; } {
      inherit systems;

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      flake.overlays.default = import ./overlay.nix;

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            config = {
              allowUnfree = true;
              cudaSupport = true;
              # Support Orin and Ada
              cudaCapabilities = if system == "aarch64-linux" then [ "8.7" ] else [ "8.9" ];
            };
            localSystem = { inherit system; };
            overlays = [ inputs.self.overlays.default ];
          };

          legacyPackages = pkgs;

          devShells.default = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  basicsr
                  einops
                  gsplat
                  imageio
                  jaxtyping
                  numpy
                  opencv4
                  pillow
                  tensorboardx
                  timm
                  torch
                  torchvision
                ]
              ))
            ];
          };

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
          };
        };
    };
}
