@echo off
cd %~dp0
cd ..
docker pull emscripten/emsdk:5.0.7
docker run --rm -v %CD%:/src emscripten/emsdk:5.0.7 node script/wasm_build.js
