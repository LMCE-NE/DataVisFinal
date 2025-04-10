"""
Chatbot Arena (side-by-side) tab with three models and image support.
Users chat with three chosen models and can upload images.
"""

import json
import time
from typing import Union, List

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    TEXT_MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)
from fastchat.serve.vision.image import Image

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 3  # Changed from 2 to 3
enable_moderation = False

# Add UI components for image handling
visible_image_column = gr.Image(visible=True)
invisible_image_column = gr.Image(visible=False)


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = [None] * num_sides

    # Select models for all three positions
    model_left = models[0] if len(models) > 0 else ""
    model_middle = models[1] if len(models) > 1 else model_left
    
    if len(models) > 2:
        weights = ([8] * 4 + [4] * 8 + [1] * 64)[: len(models) - 2]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[2:], p=weights)
    else:
        model_right = model_middle

    selector_updates = [
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_middle, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def model_a_vote(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request
):
    logger.info(f"model_a_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1, state2], "model_a_vote", [model_selector0, model_selector1, model_selector2], request
    )
    return ("",) + (disable_btn,) * 5


def model_b_vote(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request
):
    logger.info(f"model_b_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1, state2], "model_b_vote", [model_selector0, model_selector1, model_selector2], request
    )
    return ("",) + (disable_btn,) * 5


def model_c_vote(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request
):
    logger.info(f"model_c_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1, state2], "model_c_vote", [model_selector0, model_selector1, model_selector2], request
    )
    return ("",) + (disable_btn,) * 5


def tie_vote(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request
):
    logger.info(f"tie_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1, state2], "tie_vote", [model_selector0, model_selector1, model_selector2], request
    )
    return ("",) + (disable_btn,) * 5


def all_bad_vote(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request
):
    logger.info(f"all_bad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1, state2], "all_bad_vote", [model_selector0, model_selector1, model_selector2], request
    )
    return ("",) + (disable_btn,) * 5


def regenerate(state0, state1, state2, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1, state2]
    
    # Check if all models support regeneration
    all_support_regen = all(state.regen_support for state in states if state is not None)
    
    if all_support_regen:
        for i in range(num_sides):
            if states[i] is not None:
                states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 7
        )
    
    # If not all support regeneration, skip
    for state in states:
        if state is not None:
            state.skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 7


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + [""]
        + [invisible_btn] * 5
        + [disable_btn] * 2
        + [None]  # For the image preview
    )


def share_click(state0, state1, state2, model_selector0, model_selector1, model_selector2, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None and state2 is not None:
        vote_last_response(
            [state0, state1, state2], "share", [model_selector0, model_selector1, model_selector2], request
        )


# Image handling functions
def set_visible_image(chat_input):
    if isinstance(chat_input, dict) and "files" in chat_input:
        images = chat_input["files"]
        if len(images) == 0:
            return invisible_image_column
        elif len(images) > 1:
            gr.Warning(
                "We only support single image conversations. Please start a new round if you would like to chat using this image."
            )
        return visible_image_column
    return invisible_image_column


def set_invisible_image():
    return invisible_image_column


def add_image(chat_input):
    if isinstance(chat_input, dict) and "files" in chat_input:
        images = chat_input["files"]
        if len(images) > 0:
            return images[0]
    return None


def convert_images_to_conversation_format(images):
    MAX_IMAGE_SIZE_IN_MB = 5 / 1.5
    conv_images = []
    if len(images) > 0:
        conv_image = Image(url=images[0])
        conv_image.to_conversation_format(MAX_IMAGE_SIZE_IN_MB)
        conv_images.append(conv_image)
    return conv_images


def add_text(
    state0, state1, state2, model_selector0, model_selector1, model_selector2, chat_input, request: gr.Request
):
    ip = get_ip(request)
    
    # Process multimodal input
    if isinstance(chat_input, dict):
        text, images = chat_input["text"], chat_input["files"]
    else:
        text, images = chat_input, []
    
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1, state2]
    model_selectors = [model_selector0, model_selector1, model_selector2]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            if len(images) == 0:
                states[i] = State(model_selectors[i], is_vision=False)
            else:
                # Check if model supports vision
                states[i] = State(model_selectors[i], is_vision=True)

    if len(text) <= 0 and len(images) == 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + ["", None]
            + [no_change_btn,] * 7
        )

    # Process images
    conv_images = convert_images_to_conversation_format(images)
    
    # Check for image moderation
    image_flagged, csam_flagged = False, False
    if len(images) > 0:
        image_flagged, csam_flagged = image_moderation_filter(images[0])
    
    # Check text moderation
    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text = ""
    for state in states:
        if state is not None:
            all_conv_text += state.conv.get_prompt()[-1000:] + " "
    all_conv_text += "\nuser: " + text
    
    text_flagged = moderation_filter(all_conv_text, model_list)
    
    # Handle moderation results
    if text_flagged or image_flagged:
        logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
        if text_flagged and not image_flagged:
            text = TEXT_MODERATION_MSG
        elif not text_flagged and image_flagged:
            text = IMAGE_MODERATION_MSG
        else:
            text = MODERATION_MSG
            
    # Check conversation turn limit
    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [no_change_btn,] * 7
        )

    # Add text and images to each model's conversation
    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        # Check if model supports images
        if len(conv_images) > 0 and hasattr(states[i].conv, "handle_image"):
            # If there were previous images, reset the conversation
            if hasattr(states[i].conv, "get_images") and len(states[i].conv.get_images()) > 0:
                states[i].conv = get_conversation_template(states[i].model_name)
            
            user_message = (text, conv_images)
        else:
            user_message = text
            
        states[i].conv.append_message(states[i].conv.roles[0], user_message)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [disable_btn,] * 7
    )


def bot_response_multi(
    state0,
    state1,
    state2,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")

    # Check if generation should be skipped
    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state2,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
            state2.to_gradio_chatbot(),
        ) + (no_change_btn,) * 7
        return

    states = [state0, state1, state2]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    # Configure token yield rates for different models
    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemini-pro-vision",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen-vl-max-0809",
            "qwen1.5-110b-chat",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2.5-72b-instruct",
            "qwen2-72b-instruct",
            "qwen-plus-0828",
            "qwen-max-0919",
            "llama-3.1-405b-instruct-bf16",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    # Generate and yield responses
    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 7
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 5 + [enable_btn] * 2,
        [enable_btn] * 7,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.3)


def build_side_by_side_ui_named(models, vision_models=None):
    if vision_models is None:
        vision_models = []
    
    notice_markdown = f"""
# ⚔️  Triple AI Chatbot Arena: Compare 3 Models with Image Support
[GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2403.04132)

## 📜 How It Works
- Ask any question or upload an image to three chosen models and vote for the best one!
- You can chat for multiple turns until you identify a winner.
- You can upload a single image per conversation.

## 👇 Choose three models to compare
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        # Image upload area
        with gr.Row():
            multimodal_textbox = gr.MultimodalTextbox(
                file_types=["image"],
                show_label=False,
                placeholder="Enter your prompt or add image here",
                container=True,
                elem_id="input_box",
            )
            
        # Image preview
        with gr.Row():
            image_preview = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
                visible=False,
                elem_id="image_preview"
            )
            
        # Model selectors
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                    
        # Model descriptions
        with gr.Row():
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(models)} models", open=False
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        # Chatbots
        with gr.Row():
            for i in range(num_sides):
                label = f"Model {chr(65+i)}"  # "Model A", "Model B", "Model C"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id=f"chatbot_{i}",
                        height=650,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                    )

    # Voting buttons
    with gr.Row():
        model_a_btn = gr.Button(value="👈  A is best", visible=False, interactive=False)
        model_b_btn = gr.Button(value="👆  B is best", visible=False, interactive=False)
        model_c_btn = gr.Button(value="👉  C is best", visible=False, interactive=False)
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        all_bad_btn = gr.Button(value="👎  All are bad", visible=False, interactive=False)

    # Hidden regular textbox for compatibility
    textbox = gr.Textbox(
        show_label=False,
        placeholder="Hidden textbox for compatibility",
        elem_id="hidden_input_box",
        visible=False,
    )
    
    # Send button
    with gr.Row():
        send_btn = gr.Button(value="Send", variant="primary")

    # Control buttons
    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    # Parameters
    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register image handling events
    multimodal_textbox.change(
        add_image, 
        [multimodal_textbox], 
        [image_preview]
    ).then(
        set_visible_image, 
        [multimodal_textbox], 
        [image_preview]
    )

    # Register listeners for voting buttons
    btn_list = [
        model_a_btn,
        model_b_btn,
        model_c_btn,
        tie_btn,
        all_bad_btn,
        regenerate_btn,
        clear_btn,
    ]
    
    model_a_btn.click(
        model_a_vote,
        states + model_selectors,
        [textbox, model_a_btn, model_b_btn, model_c_btn, tie_btn, all_bad_btn],
    )
    model_b_btn.click(
        model_b_vote,
        states + model_selectors,
        [textbox, model_a_btn, model_b_btn, model_c_btn, tie_btn, all_bad_btn],
    )
    model_c_btn.click(
        model_c_vote,
        states + model_selectors,
        [textbox, model_a_btn, model_b_btn, model_c_btn, tie_btn, all_bad_btn],
    )
    tie_btn.click(
        tie_vote,
        states + model_selectors,
        [textbox, model_a_btn, model_b_btn, model_c_btn, tie_btn, all_bad_btn],
    )
    all_bad_btn.click(
        all_bad_vote,
        states + model_selectors,
        [textbox, model_a_btn, model_b_btn, model_c_btn, tie_btn, all_bad_btn],
    )
    
    # Register regenerate and clear buttons
    regenerate_btn.click(
        regenerate, 
        states, 
        states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    
    clear_btn.click(
        clear_history, 
        None, 
        states + chatbots + [textbox] + btn_list + [image_preview]
    )

    # Register share button
    share_js = """
function (a, b, c, d, e, f) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena-triple.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d, e, f];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    # Register model selector changes
    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, 
            None, 
            states + chatbots + [textbox] + btn_list + [image_preview]
        )

    # Register text input events
    multimodal_textbox.submit(
        add_text,
        states + model_selectors + [multimodal_textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        set_invisible_image,
        None,
        [image_preview]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    
    send_btn.click(
        add_text,
        states + model_selectors + [multimodal_textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        set_invisible_image,
        None,
        [image_preview]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors


def load_demo(models, url_params, enable_moderation=False, vision_models=None):
    """
    Load the demo for the gradio UI.
    """
    set_global_vars_named(enable_moderation)
    return load_demo_side_by_side_named(models, url_params)


def get_conversation_template_for_vision(model_name):
    """
    Helper function to get conversation templates that support vision
    """
    conv = get_conversation_template(model_name)
    
    # Monkey patch get_images method if it doesn't exist
    if not hasattr(conv, "get_images"):
        def get_images(self):
            return []
        conv.get_images = get_images.__get__(conv)
    
    return conv