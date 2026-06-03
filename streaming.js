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
          "powershell -NoProfile -Command \"Expand-Archive -Path '{{path.resolve(cwd, 'app', 'unity_capture.zip')}}' -DestinationPath '{{path.resolve(cwd, 'app', 'driver')}}' -Force\""
        ]
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: "fs.rm",
      params: {
        path: "app/unity_capture.zip"
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: "shell.run",
      params: {
        message: [
          "powershell -NoProfile -Command \"Start-Process regsvr32.exe -ArgumentList '/s \\\"{{path.resolve(cwd, 'app', 'driver', 'UnityCapture-master', 'Install', 'UnityCaptureFilter32.dll')}}\\\"' -Verb RunAs -WindowStyle Hidden -Wait; Start-Process regsvr32.exe -ArgumentList '/s \\\"{{path.resolve(cwd, 'app', 'driver', 'UnityCapture-master', 'Install', 'UnityCaptureFilter64.dll')}}\\\"' -Verb RunAs -WindowStyle Hidden -Wait\""
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Streaming Service installed successfully."
      }
    }
  ]
}