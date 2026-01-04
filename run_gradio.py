"""
Simple script to run the Gradio interface for HY-MT1.5
"""
from app import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

