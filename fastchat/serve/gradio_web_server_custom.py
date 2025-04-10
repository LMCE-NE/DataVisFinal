"""
Clean implementation of a 3-model comparison interface with no terms of service.
"""

import argparse
import gradio as gr

from fastchat.serve.gradio_global_state import Context
from fastchat.serve.gradio_web_server import (
    set_global_vars,
    get_model_list,
    get_ip,
)
from fastchat.utils import (
    build_logger,
    parse_gradio_auth_creds,
)

from fastchat.serve.gradio_block_arena_three_vision import (
    build_side_by_side_ui_named as build_three_model_ui,
    load_demo as load_three_model_demo,
    set_global_vars_named as set_three_model_vars,
)

logger = build_logger("triple_model_interface", "triple_model_interface.log")

# Custom CSS to hide any unwanted elements
custom_css = """
/* Hide any terms of service, notices, or undesired content */
#notice_markdown, 
#ack_markdown,
.gradio-container .terms, 
.gradio-container .acknowledgments,
.gradio-container .survey-link,
.gradio-container .survey,
.gradio-container .tos,
.gradio-container .terms-of-service,
.tos-container {
    display: none !important;
}

/* Custom styling for the main title */
#custom_title {
    margin-bottom: 20px;
    text-align: center;
}

/* Simple acknowledgment styling */
#simple_acknowledgment {
    margin-top: 20px;
    font-size: 0.9em;
    text-align: center;
    color: #555;
}
"""

def load_demo(context: Context, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}")

    # Load our 3-model UI only
    three_model_updates = load_three_model_demo(context.models, {})
    
    # Create tabs list with only our UI
    tabs_list = [gr.Tabs(selected=0)] + three_model_updates

    return tabs_list


def build_demo(context: Context):
    """Build the demo with only the 3-model comparison UI"""
    
    # Basic JS for window parameters without terms
    load_js = """
function() {
    function load() {
        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        const params = Object.fromEntries(urlParams.entries());
        return params;
    }
    return load();
}
    """
    
    head_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
"""

    # Create a custom theme to override any built-in styles
    custom_theme = gr.themes.Default().set(
        body_text_color="#212121",
        body_background_fill="#FFFFFF",
        button_primary_background_fill="#2196F3",
    )

    with gr.Blocks(
        title="Triple Model Comparison with Image Support",
        theme=custom_theme,
        css=custom_css,
        head=head_js,
    ) as demo:
        # Simple title
        gr.Markdown("# Triple Model Comparison with Image Support", elem_id="custom_title")
        
        # Only our 3-model UI
        with gr.Tabs() as inner_tabs:
            with gr.Tab("Compare 3 Models", id=0):
                three_model_list = build_three_model_ui(
                    context.models, vision_models=context.vision_models
                )
        
        # Simple acknowledgment
        gr.Markdown("Built on [FastChat](https://github.com/lm-sys/FastChat)", elem_id="simple_acknowledgment")

        context_state = gr.State(context)

        demo.load(
            load_demo,
            [context_state],
            [inner_tabs] + three_model_list,
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true", help="Generate a public link")
    parser.add_argument(
        "--controller-url", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--vision-arena", action="store_true", help="Enable vision models")
    parser.add_argument("--register-api-endpoint-file", type=str)
    parser.add_argument("--gradio-auth-path", type=str)
    parser.add_argument("--gradio-root-path", type=str)
    parser.add_argument("--use-remote-storage", action="store_true", default=False)
    parser.add_argument("--password", type=str)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_three_model_vars(args.moderate)
    
    # Get model lists
    text_models, all_text_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=False,
    )

    vision_models, all_vision_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=True,
    )

    # Combine models
    models = text_models + [model for model in vision_models if model not in text_models]
    all_models = all_text_models + [model for model in all_vision_models if model not in all_text_models]
    
    # Create context
    context = Context(
        text_models,
        all_text_models,
        vision_models,
        all_vision_models,
        models,
        all_models,
    )

    # Set authentication
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(context)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
        show_api=False,
    )