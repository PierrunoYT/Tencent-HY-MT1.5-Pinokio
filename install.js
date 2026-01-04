module.exports = {
  run: [
    // Install PyTorch with CUDA support first
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: false   // Not needed for translation models
        }
      }
    },
    // Install HY-MT1.5 dependencies from requirements.txt
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch HY-MT1.5. Models will be downloaded automatically from Hugging Face on first use."
      }
    }
  ]
}
