# COLMAP-SLAM v1.0

## Cloning the repository
This repository includes [colmap-odometry](https://github.com/3DOM-FBK/colmap-odometry) as a git submodule in `third_party/colmap-odometry`. Clone with `--recursive` to fetch it too:

```bash
git clone --recursive https://github.com/lcmrl/cslam.git
```

If you already cloned without `--recursive`, initialize the submodule with:

```bash
git submodule update --init --recursive
```

For compatibility, the submodule is pinned to a specific commit of colmap-odometry:

| Submodule | Commit | Date |
|-----------|--------|------|
| colmap-odometry | [`fdcfbce`](https://github.com/3DOM-FBK/colmap-odometry/commit/fdcfbce53e85bcd6e139fa9b252e1ba16eadff98) | 2026-11-10 |

To check that you have the pinned version, run:

```bash
git submodule status
# expected: fdcfbce53e85bcd6e139fa9b252e1ba16eadff98 third_party/colmap-odometry
```

If the hash is different, or has a `+` in front of it, restore the pinned commit with `git submodule update --init --recursive`.

`git submodule update --init --recursive` always checks out this exact commit, so later upstream changes won't affect this repository. Do **not** use `git submodule update --remote` unless you intend to move to a newer, possibly incompatible, version. If you do, check that everything still works before committing the new pointer.

## Installation
For installing colmap-slam, we recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then, you can install colmap-slam using uv:

```bash
uv pip install -e .
```

This command will install the package in editable mode, allowing you to modify the source code and see changes immediately without needing to reinstall. If you want to use colmap-slam as a non-editable library, you can also install it without the `-e` flag.

Install `pytorch` from https://pytorch.org/ to have GPU support.

## Running the code



### Reference

Morelli, L., Ioli, F., Beber, R., Menna, F., Remondino, F. and Vitti, A., 2023. COLMAP-SLAM: A FRAMEWORK FOR VISUAL ODOMETRY. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48, pp.317-324.