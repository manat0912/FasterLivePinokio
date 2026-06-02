module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "cmd /d /c build_trt.bat"
      }
    }
  ]
}