module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      when: "{{platform === 'win32'}}",
      method: "fs.download",
      params: {
        url: "https://github.com/schellingb/UnityCapture/archive/refs/heads/master.zip",
        path: "app/unity_capture.zip"
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: "shell.run",
      params: {
        message: [
          "powershell -Command \"Expand-Archive -Path app\\unity_capture.zip -DestinationPath app\\driver -Force\"",
          "del app\\unity_capture.zip",
          "powershell -Command \"Start-Process regsvr32.exe -ArgumentList '/s \\\"{{path.resolve(cwd, 'app', 'driver', 'UnityCapture-master', 'Install', 'UnityCaptureFilter.dll')}}\\\"' -Verb RunAs -Wait\""
        ]
      }
    }
  ]
}