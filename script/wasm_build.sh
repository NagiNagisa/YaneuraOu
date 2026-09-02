#!/bin/bash
pushd `dirname $0`
pushd ../
docker pull emscripten/emsdk:5.0.7
docker run --rm -v ${PWD}:/src emscripten/emsdk:5.0.7 node script/wasm_build.js
popd
popd
