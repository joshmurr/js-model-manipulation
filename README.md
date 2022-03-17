## Issues

- TensorflowJS (or more likely WebGL) apparently has an issue with NVIDIA _and_ Firefox. You'll see the following errors:

```
Uncaught (in promise) Error: Failed to compile fragment shader.

...

Couldn't parse line number in error: 0(32) : error C7532: global function uintBitsToFloat requires "#version 330" or later
0(32) : error C0000: ... or #extension GL_ARB_shader_bit_encoding : enable
0(32) : error C0000: ... or #extension GL_ARB_gpu_shader5 : enable
```

- but currently it works ok in Chrome.
