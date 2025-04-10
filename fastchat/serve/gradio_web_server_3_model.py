"""
The gradio demo server with multiple tabs.
Modified to show 3-model comparison with image support as the primary interface.
"""

import argparse
import gradio as gr

from fastchat.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from fastchat.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from fastchat.serve.gradio_global_state import Context

from fastchat.serve.gradio_web_server_custom import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_about,
    get_model_list,
    load_demo_single,
    get_ip,
)
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    parse_gradio_auth_creds,
)

from fastchat.serve.gradio_block_arena_three_vision import (
    build_side_by_side_ui_named as build_three_model_ui,
    load_demo as load_three_model_demo,
    set_global_vars_named as set_three_model_vars,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(context: Context, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {request.query_params}")

    inner_selected = 0  # Always start with our custom 3-model UI
    
    if args.model_list_mode == "reload":
        context.text_models, context.all_text_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=False,
        )

        context.vision_models, context.all_vision_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=True,
        )

    # Load our custom 3-model UI
    three_model_updates = load_three_model_demo(context.models, request.query_params)
    

    tabs_list = (
        [gr.Tabs(selected=inner_selected)]
        + three_model_updates  
    )

    return tabs_list


def build_demo(
    context: Context, elo_results_file: str, leaderboard_table_file, arena_hard_table
):
    load_js = get_window_url_params_js

    head_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
"""
    if args.ga_id is not None:
        head_js += f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={args.ga_id}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());

gtag('config', '{args.ga_id}');
window.__gradio_mode__ = "app";
</script>
        """
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="Data Visualization Tester",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
        head=head_js,
    ) as demo:
        # Add custom title at the top
        gr.Markdown("# Data Visualization Tester", elem_id="custom_title")
        
        with gr.Tabs() as inner_tabs:
            # Our custom 3-model UI is the first and only visible tab
            with gr.Tab("Compare 3 Models in Data Visualization Tasks", id=0) as triple_arena_tab:
                triple_arena_tab.select(None, None, None, js=load_js)
                three_model_list = build_three_model_ui(
                    context.models, vision_models=context.vision_models
                )

            demo_tabs = (
                [inner_tabs]
                + three_model_list
            )

            with gr.Tab("ℹ️ About", id=6, visible=False):
                build_about()
        
        # Add simple acknowledgment at the bottom
        gr.Markdown("Built with [FastChat](https://github.com/lm-sys/FastChat) and Gradio", elem_id="simple_acknowledgment")

        context_state = gr.State(context)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        demo.load(
            load_demo,
            [context_state],
            demo_tabs,
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--vision-arena", action="store_true", help="Show tabs for vision arena."
    )
    parser.add_argument(
        "--random-questions", type=str, help="Load random questions from a JSON file"
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or \
              more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--arena-hard-table", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a \
              reverse-proxy or at a custom URL path prefix",
    )
    parser.add_argument(
        "--ga-id",
        type=str,
        help="the Google Analytics ID",
        default=None,
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Set the password for the gradio web server",
    )
    parser.add_argument(
        "--show-visualizer",
        action="store_true",
        default=False,
        help="Show the Data Visualizer tab",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    set_three_model_vars(args.moderate)  # Set vars for our custom 3-model UI
    
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

    models = text_models + [
        model for model in vision_models if model not in text_models
    ]
    all_models = all_text_models + [
        model for model in all_vision_models if model not in all_text_models
    ]
    context = Context(
        text_models,
        all_text_models,
        vision_models,
        all_vision_models,
        models,
        all_models,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(
        context,
        args.elo_results_file,
        args.leaderboard_table_file,
        args.arena_hard_table,
    )
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