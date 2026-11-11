"""
DeepSecure AI — Multimodal Deepfake Detection System
====================================================
Minimal Gradio interface for image, video, and audio deepfake detection.
"""

import os
import re
import html
import numpy as np
import gradio as gr

# Ensure ffmpeg/ffprobe binary paths are available in PATH for audio decoding
POSSIBLE_FFMPEG_PATHS = [
    r"C:\Users\kumar\ffmpeg\ffmpeg-9.0.2-essentials_build\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin"
]
for p in POSSIBLE_FFMPEG_PATHS:
    if os.path.exists(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

# Load backend inference module
import inference_2 as inference

# Load custom CSS
STYLE_FILE = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(STYLE_FILE):
    with open(STYLE_FILE, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()
else:
    CUSTOM_CSS = ""

# ---------------------------------------------------------------------------
# HTML Card Builders & UI Components
# ---------------------------------------------------------------------------

HEADER_HTML = """
<header class="page-header">
  <h1>DeepSecure AI</h1>
  <p>Upload an image, video, or audio file to analyze whether it is AI-generated or manipulated.</p>
</header>
"""

FOOTER_HTML = """
<p class="ds-footer-disclaimer">AI predictions are probabilistic and should not be considered absolute proof of authenticity.</p>
"""

def render_idle_card(modality: str) -> str:
    """Show a quiet prompt before a file is analyzed."""
    return f"""
    <div class="result-state result-idle">Your {modality.lower()} analysis result will appear here.</div>
    """

def render_result_card(raw_output: str, modality: str) -> str:
    """Show the prediction and only evidence present in the inference output."""
    if not raw_output:
        return render_idle_card(modality)

    clean_text = raw_output.strip()
    upper = clean_text.upper()
    is_warning = "PLEASE UPLOAD" in upper or "ERROR" in upper or upper.startswith("NO ")
    if is_warning:
        return f"""
        <section class="result-state result-neutral model-response">
          <h2>Model Response</h2>
          <p>{html.escape(clean_text)}</p>
        </section>
        """

    prediction_match = re.search(r"\bis\s+(REAL|FAKE)\b", clean_text, re.IGNORECASE)
    if not prediction_match:
        prediction_match = re.search(r"\b(REAL|FAKE)\b", clean_text, re.IGNORECASE)
    if not prediction_match:
        return f"""
        <section class="result-state result-neutral model-response">
          <h2>Model Response</h2>
          <p>{html.escape(clean_text)}</p>
        </section>
        """

    prediction = prediction_match.group(1).upper()
    result_class = "real" if prediction == "REAL" else "fake"
    confidence_match = re.search(
        r"confidence\s+score\s+is:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        clean_text,
        re.IGNORECASE,
    )
    confidence_html = ""
    if confidence_match:
        confidence_html = f"""
        <div class="response-item">
          <span class="response-label">Confidence score</span>
          <strong>{html.escape(confidence_match.group(1))}%</strong>
        </div>
        """

    explanations = {
        "Image": (
            "The image classifier bases this result on learned visual patterns, but it does not identify which image regions or features influenced the prediction.",
            "The confidence shown is the class score returned by the model. Feature-level evidence is not included in its output.",
        ),
        "Video": (
            "The model does not identify which frames or visual cues contributed most to this prediction.",
            "The video path averages class scores from three evenly spaced frames; the confidence shown is the selected class's averaged score.",
        ),
        "Audio": (
            "The audio model returns a class without identifying specific acoustic patterns that led to it.",
            "The current audio inference output includes no confidence score or feature-level explanation.",
        ),
    }
    reason, analysis = explanations.get(
        modality,
        ("The model does not expose feature-level evidence for this prediction.", "The explanation is limited to information returned by inference."),
    )

    return f"""
    <section class="result-state result-{result_class} model-response">
      <h2>Model Response</h2>
      <div class="response-item">
        <span class="response-label">Prediction</span>
        <strong>{html.escape(modality)} is {prediction}</strong>
      </div>
      {confidence_html}
      <div class="response-explanation">
        <h3>Reason for classification</h3>
        <p>{reason}</p>
        <h3>Analysis</h3>
        <p>{analysis}</p>
      </div>
      <details class="raw-model-output">
        <summary>Original model response</summary>
        <p>{html.escape(clean_text)}</p>
      </details>
    </section>
    """
# ---------------------------------------------------------------------------
# Backend Prediction Handlers (Preserving Original Inference Functions)
# ---------------------------------------------------------------------------

def handle_image_analysis(img):
    if img is None:
        msg = "Please upload or select an image file to analyze."
        return render_result_card(msg, "Image"), msg
    
    # Call original inference function
    raw_res = inference.deepfakes_image_predict(img)
    return render_result_card(raw_res, "Image"), raw_res

def handle_video_analysis(video_path):
    if not video_path:
        msg = "Please upload or select a video file to analyze."
        return render_result_card(msg, "Video"), msg
    
    # Call original inference function
    raw_res = inference.deepfakes_video_predict(video_path)
    return render_result_card(raw_res, "Video"), raw_res

def handle_audio_analysis(audio_tuple):
    if audio_tuple is None:
        msg = "Please upload or select an audio file to analyze."
        return render_result_card(msg, "Audio"), msg
    
    # In Gradio 6, gr.Audio returns (sample_rate, numpy_data)
    # The inference module unpacks `x, _ = input_audio` expecting x to be the numpy array
    if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
        a, b = audio_tuple
        if isinstance(a, (int, np.integer)) and isinstance(b, np.ndarray):
            # Pass (data, sr) so x receives the numpy audio array
            processed_tuple = (b, a)
        else:
            processed_tuple = audio_tuple
    else:
        processed_tuple = audio_tuple

    # Call original inference function
    raw_res = inference.deepfakes_spec_predict(processed_tuple)
    return render_result_card(raw_res, "Audio"), raw_res

# ---------------------------------------------------------------------------
# Gradio Interface Construction
# ---------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#0b0d12",
    body_background_fill_dark="#0b0d12",
    block_background_fill="#14171e",
    block_background_fill_dark="#14171e",
    block_border_color="#282d36",
    block_border_color_dark="#282d36",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_dark="#6366f1",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff"
)

with gr.Blocks(title="DeepSecure AI — Multimodal Deepfake Detection System") as app:
    if CUSTOM_CSS:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    theme_toggle = gr.Button("☾", elem_id="theme-toggle", elem_classes=["ds-theme-toggle"])
    gr.HTML(HEADER_HTML)

    app.load(
        fn=None,
        js="""() => {
          const mode = sessionStorage.getItem('deepsecure-theme') || 'dark';
          document.documentElement.dataset.theme = mode;
          const button = document.querySelector('#theme-toggle button');
          if (button) {
            button.textContent = mode === 'light' ? '☼' : '☾';
            button.setAttribute('aria-label', mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
          }
        }"""
    )
    theme_toggle.click(
        fn=None,
        js="""() => {
          const root = document.documentElement;
          const mode = root.dataset.theme === 'light' ? 'dark' : 'light';
          root.dataset.theme = mode;
          sessionStorage.setItem('deepsecure-theme', mode);
          const button = document.querySelector('#theme-toggle button');
          if (button) {
            button.textContent = mode === 'light' ? '☼' : '☾';
            button.setAttribute('aria-label', mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
          }
        }"""
    )

    with gr.Tabs(elem_classes=["ds-tabs-container"]):
        with gr.Tab("Image", id="tab_image"):
            with gr.Column(elem_classes=["workspace"]):
                image_input = gr.Image(
                    label="Drop your image here or browse",
                    type="numpy",
                    elem_classes=["media-input-box"],
                    sources=["upload", "clipboard"]
                )
                gr.Markdown("JPG, PNG, WEBP", elem_classes=["format-hint"])
                with gr.Row(elem_classes=["action-row"]):
                    image_clear_btn = gr.Button("Reset", variant="secondary", elem_classes=["ds-action-btn-secondary"])
                    image_analyze_btn = gr.Button("Analyze Image", variant="primary", elem_classes=["ds-action-btn-primary"])
                with gr.Accordion("Try a sample", open=False):
                    gr.Examples(
                        examples=[["images/lady.jpg"], ["images/fake_image.jpg"]],
                        inputs=image_input,
                        label="Demonstration images",
                        cache_examples=False
                    )
                image_result_html = gr.HTML(value=render_idle_card("Image"), elem_id="image-result-html")
                with gr.Accordion("Technical details", open=False):
                    image_raw_code = gr.Textbox(label="Model response", lines=3, interactive=False)

            image_analyze_btn.click(
                fn=handle_image_analysis,
                inputs=[image_input],
                outputs=[image_result_html, image_raw_code],
                show_progress="full"
            )
            image_clear_btn.click(
                fn=lambda: (None, render_idle_card("Image"), ""),
                outputs=[image_input, image_result_html, image_raw_code]
            )

        with gr.Tab("Video", id="tab_video"):
            with gr.Column(elem_classes=["workspace"]):
                video_input = gr.Video(
                    label="Drop your video here or browse",
                    elem_classes=["media-input-box"],
                    sources=["upload"]
                )
                gr.Markdown("MP4, AVI, MOV, WEBM", elem_classes=["format-hint"])
                with gr.Row(elem_classes=["action-row"]):
                    video_clear_btn = gr.Button("Reset", variant="secondary", elem_classes=["ds-action-btn-secondary"])
                    video_analyze_btn = gr.Button("Analyze Video", variant="primary", elem_classes=["ds-action-btn-primary"])
                with gr.Accordion("Try a sample", open=False):
                    gr.Examples(
                        examples=[["videos/real-1.mp4"], ["videos/celeb_synthesis.mp4"], ["videos/aaa.mp4"]],
                        inputs=video_input,
                        label="Demonstration videos",
                        cache_examples=False
                    )
                video_result_html = gr.HTML(value=render_idle_card("Video"), elem_id="video-result-html")
                with gr.Accordion("Technical details", open=False):
                    video_raw_code = gr.Textbox(label="Model response", lines=3, interactive=False)

            video_analyze_btn.click(
                fn=handle_video_analysis,
                inputs=[video_input],
                outputs=[video_result_html, video_raw_code],
                show_progress="full"
            )
            video_clear_btn.click(
                fn=lambda: (None, render_idle_card("Video"), ""),
                outputs=[video_input, video_result_html, video_raw_code]
            )

        with gr.Tab("Audio", id="tab_audio"):
            with gr.Column(elem_classes=["workspace"]):
                audio_input = gr.Audio(
                    label="Drop your audio here or browse",
                    type="numpy",
                    elem_classes=["media-input-box"],
                    sources=["upload", "microphone"]
                )
                gr.Markdown("FLAC, WAV, MP3", elem_classes=["format-hint"])
                with gr.Row(elem_classes=["action-row"]):
                    audio_clear_btn = gr.Button("Reset", variant="secondary", elem_classes=["ds-action-btn-secondary"])
                    audio_analyze_btn = gr.Button("Analyze Audio", variant="primary", elem_classes=["ds-action-btn-primary"])
                with gr.Accordion("Try a sample", open=False):
                    gr.Examples(
                        examples=[["audios/DF_E_2000027.flac"], ["audios/DF_E_2000031.flac"]],
                        inputs=audio_input,
                        label="Demonstration audio",
                        cache_examples=False
                    )
                audio_result_html = gr.HTML(value=render_idle_card("Audio"), elem_id="audio-result-html")
                with gr.Accordion("Technical details", open=False):
                    audio_raw_code = gr.Textbox(label="Model response", lines=3, interactive=False)

            audio_analyze_btn.click(
                fn=handle_audio_analysis,
                inputs=[audio_input],
                outputs=[audio_result_html, audio_raw_code],
                show_progress="full"
            )
            audio_clear_btn.click(
                fn=lambda: (None, render_idle_card("Audio"), ""),
                outputs=[audio_input, audio_result_html, audio_raw_code]
            )

    gr.HTML(FOOTER_HTML)
if __name__ == "__main__":
    app.launch(share=False, theme=theme, css=CUSTOM_CSS)
