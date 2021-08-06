
bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip3 install --no-deps  --force-reinstall artifacts/tensorflow_recommenders_addons*.whl
