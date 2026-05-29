import pefile
pe = pefile.PE('checkpoints/liveportrait_onnx/grid_sample_3d_plugin.dll')
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    print(entry.dll)
