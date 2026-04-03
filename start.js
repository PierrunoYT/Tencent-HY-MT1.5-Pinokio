module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          event: "/http:\/\/[0-9.:]+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "HY-MT1.5 is running! Click 'Open Web UI' to start translating between 33 languages."
      }
    }
  ]
}
