# HY-MT1.5 Pinokio

🌐 **Hunyuan Translation Model Version 1.5** - A Gradio web interface for the HY-MT1.5 translation models, packaged for Pinokio.

## Overview

HY-MT1.5 includes two translation models:
- **HY-MT1.5-1.8B**: Fast, efficient model suitable for edge devices and real-time translation
- **HY-MT1.5-7B**: High-accuracy model optimized for explanatory translation and mixed-language scenarios

Both models support mutual translation across **33 languages** and include features like:
- Terminology intervention
- Contextual translation
- Formatted translation (preserving formatting tags)

## Features

- 🌍 **33 Languages Support**: Translate between Chinese, English, French, Spanish, Japanese, Korean, and 27 more languages
- 🎯 **Multiple Translation Modes**:
  - **Basic**: Standard translation
  - **Terminology**: Translation with custom terminology guide
  - **Contextual**: Translation with context for better accuracy
  - **Formatted**: Preserve formatting tags (for Chinese translation)
- ⚡ **Model Selection**: Choose between 1.8B (faster) or 7B (more accurate) models
- 🎛️ **Customizable Parameters**: Adjust temperature, top-p, top-k, and repetition penalty
- 🖥️ **Web Interface**: User-friendly Gradio interface accessible via browser

## Installation

### Using Pinokio

1. **Install** the app through Pinokio
2. Click **"Start"** to launch the Gradio interface
3. Open the web UI from the Pinokio interface

### Manual Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the interface**:
```bash
python app.py
```

Or use the helper script:
```bash
python run_gradio.py
```

4. **Access the interface**:
Open your browser and navigate to `http://localhost:7860`

## Usage

### Basic Translation

1. Select the **source language** and **target language** from the dropdowns
2. Choose your preferred **model** (1.8B or 7B)
3. Enter the text to translate in the **Source Text** box
4. Click **"Translate"**
5. View the translation in the **Translation** box

### Terminology Mode

1. Select **"terminology"** from Translation Mode
2. Enter terminology guide in the format: `source_term -> target_term`
   - Example: `AI -> 人工智能`
3. Enter your text to translate
4. Click **"Translate"**

### Contextual Mode

1. Select **"contextual"** from Translation Mode
2. Enter context that helps with translation
3. Enter your text to translate
4. Click **"Translate"**

### Formatted Mode

1. Select **"formatted"** from Translation Mode
2. Enter text with `<sn></sn>` tags to preserve formatting
3. Target language should be Chinese
4. Click **"Translate"**

## Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese | zh | English | en |
| French | fr | Portuguese | pt |
| Spanish | es | Japanese | ja |
| Turkish | tr | Russian | ru |
| Arabic | ar | Korean | ko |
| Thai | th | Italian | it |
| German | de | Vietnamese | vi |
| Malay | ms | Indonesian | id |
| Filipino | tl | Hindi | hi |
| Traditional Chinese | zh-Hant | Polish | pl |
| Czech | cs | Dutch | nl |
| Khmer | km | Burmese | my |
| Persian | fa | Gujarati | gu |
| Urdu | ur | Telugu | te |
| Marathi | mr | Hebrew | he |
| Bengali | bn | Tamil | ta |
| Ukrainian | uk | Tibetan | bo |
| Kazakh | kk | Mongolian | mn |
| Uyghur | ug | Cantonese | yue |

## Model Links

- **HY-MT1.5-1.8B**: [Hugging Face](https://huggingface.co/tencent/HY-MT1.5-1.8B)
- **HY-MT1.5-7B**: [Hugging Face](https://huggingface.co/tencent/HY-MT1.5-7B)

Models are automatically downloaded from Hugging Face on first use.

## Generation Parameters

Recommended parameters (pre-set in the interface):
- **Temperature**: 0.7
- **Top-p**: 0.6
- **Top-k**: 20
- **Repetition Penalty**: 1.05

You can adjust these parameters in the interface to fine-tune translation quality.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster inference)
- Transformers 4.56.0
- Gradio 4.0+

## Command Line Options

```bash
python app.py --help
```

Options:
- `--share`: Create a public Gradio link
- `--server-name`: Server hostname (default: 0.0.0.0)
- `--server-port`: Server port (default: 7860)

## Pinokio Commands

- **Install**: Sets up the Python environment and installs dependencies
- **Start**: Launches the Gradio web interface
- **Update**: Updates the repository
- **Reset**: Removes the virtual environment
- **Save Disk Space**: Deduplicates redundant library files

## Notes

- First translation may take longer as the model loads
- GPU is recommended for faster inference
- Models will be automatically downloaded from Hugging Face on first use
- The 1.8B model is faster and suitable for real-time scenarios
- The 7B model provides higher accuracy for complex translations

## Technical Details

The interface uses the Hugging Face Transformers library to load and run the HY-MT1.5 models. Translation prompts follow the official templates provided in the model documentation:

- **ZH<=>XX**: Uses Chinese prompt template
- **XX<=>XX (non-Chinese)**: Uses English prompt template
- **Terminology**: Includes terminology reference
- **Contextual**: Includes context information
- **Formatted**: Preserves formatting tags

## License

Please refer to the original HY-MT1.5 model license on Hugging Face.

## References

- [HY-MT1.5 on Hugging Face](https://huggingface.co/collections/tencent/hy-mt15)
- [Official Website](https://hunyuan.tencent.com)
- [GitHub Repository](https://github.com/Tencent-Hunyuan/HY-MT)

## Contact

For questions about the HY-MT1.5 models, contact: hunyuan_opensource@tencent.com

