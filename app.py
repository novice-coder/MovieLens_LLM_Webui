import os
import argparse
import time
import pickle
import sqlite3 as db
from sqlite3 import Error
from datetime import datetime
from typing import Iterator
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from distutils.util import strtobool

from llama2_wrapper import LLAMA2_WRAPPER

import logging

from prompts.utils import PromtsContainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument(
        "--backend_type",
        type=str,
        default="",
        help="Backend options: llama.cpp, gptq, transformers",
    )
    parser.add_argument(
        "--load_in_8bit",
        type=bool,
        default=False,
        help="Whether to use bitsandbytes 8 bit.",
    )
    parser.add_argument(
        "--share",
        type=bool,
        default=False,
        help="Whether to share public for gradio.",
    )
    args = parser.parse_args()

    load_dotenv()

    DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
    MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
    MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4000))

    DATA_PATH = os.getenv("DATA_PATH")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    MODEL_PATH = os.getenv("MODEL_PATH")
    assert MODEL_PATH is not None, f"MODEL_PATH is required, got: {MODEL_PATH}"
    BACKEND_TYPE = os.getenv("BACKEND_TYPE")
    assert BACKEND_TYPE is not None, f"BACKEND_TYPE is required, got: {BACKEND_TYPE}"

    LOAD_IN_8BIT = bool(strtobool(os.getenv("LOAD_IN_8BIT", "True")))

    if args.model_path != "":
        MODEL_PATH = args.model_path
    if args.backend_type != "":
        BACKEND_TYPE = args.backend_type
    if args.load_in_8bit:
        LOAD_IN_8BIT = True

    llama2_wrapper = LLAMA2_WRAPPER(
        model_path=MODEL_PATH,
        backend_type=BACKEND_TYPE,
        max_tokens=MAX_INPUT_TOKEN_LENGTH,
        load_in_8bit=LOAD_IN_8BIT,
        # verbose=True,
    )

    DESCRIPTION = """
    # MovieLens Study
    #### Instructions:
    1: You need to enter your Experiment ID as shown in the MovieLens homepage.  
    2: Select the appropriate scenario you are currently exploring from the drop-down below.  
    
    We suggest you reuse the current tab for all the scenarios, and it is **important to `clear` your conversation history before beginning a new scenario**.
    #### Good to know:
    - Your request may be queued to accomodate multiple users, so we request you to be patient. If you're part of a queue, your position will be indicated at the top-right corner of the "MovieLens" chatbot window shown below.
    - The model may output erroneous text sometimes. You may ask it to re-try or ask a new question. We appreciate your patience.

    Feel free to converse with the chatbot and tailor your recommendations. We hope you enjoy!
    """
    DESCRIPTION2 = """
    - Supporting models: [Llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML), [CodeLlama](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GPTQ) ...
    - Supporting model backends: [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), [llama.cpp](https://github.com/ggerganov/llama.cpp)
    """

    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return "", message

    def display_input(
        message: str, history: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        history.append((message, ""))
        return history

    def delete_prev_fn(
        history: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], str]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        return history, message or ""
    
    def load_userIds() -> list:
        uids = None
        try:
            conn = db.connect(database_path)
            cur = conn.cursor()

            cur.execute("SELECT exptUserId FROM usr_prompts")
            rows = cur.fetchall()
            uids = [row[0] for row in rows]
        except Error as e:
            logging.error("Failed to load the user prompts map.")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
        return uids
    
    def archive_prompt_history(
            userId: str
    ) -> str:
        
        curr_date = datetime.today().strftime('%Y_%m_%d')
        timestamp = str(time.time())
        try:
            conn = db.connect(database_path)
            cur = conn.cursor()
            update_query = """UPDATE usr_interactions SET tstamp = ? where exptUserId = ? AND tstamp = ?"""
            data = [timestamp, userId, curr_date]
            cur.execute(update_query, data)
            conn.commit()
        except Error as e:
            logging.error(f"Failed archiving prompt history for user: {userId}.")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
        
        # curr_date = datetime.today().strftime('%Y_%m_%d')
        # old_filename = f"{userId}_prompt_history_{curr_date}.txt"
        # history_path = os.path.join(DATA_PATH, HISTORY_PATH)
        # old_filepath = os.path.join(history_path, old_filename)
        # history_file = Path(old_filepath)
        # if history_file.exists():
        #     try:
        #         timestamp = int(time.time())
        #         new_filename = "{0}_prompt_history_{1}.txt".format(userId, str(timestamp))
        #         new_filepath = os.path.join(history_path, new_filename)
        #         os.rename(old_filepath, new_filepath)
        #     except:
        #         logging.error(f"Failed renaming previous prompt history file for user: {userId}. New prompt history may override old one.")
        # else:
        #     logging.info(f"User previous history file doesn't exist for user: {userId}. Skipping any action.")

    
    def save_prompt_history(
            userId: str, scenario: str, system_prompt: str, history: list[tuple[str, str]]
    ) -> None:
        history_txt = "SYSTEM PROMPT:\n" + system_prompt
        for question, response in history:
            history_txt += "\n\n"
            history_txt += "USER QUERY:\n" + question
            history_txt += "\n\n"
            history_txt += "MODEL RESPONSE:\n" + response
        
        curr_date = datetime.today().strftime('%Y_%m_%d')
        timestamp = str(time.time())
        try:
            conn = db.connect(database_path)
            cur = conn.cursor()
            update_query = """INSERT OR REPLACE INTO usr_interactions VALUES (?, ?, ?, ?)"""
            data = [userId, scenario, timestamp, history_txt]
            cur.execute(update_query, data)
            conn.commit()
        except Error as e:
            logging.error(f"Failed saving prompt history for user: {userId}.")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
        
        # history_path = os.path.join(DATA_PATH, HISTORY_PATH)
        # try:
        #     if not os.path.exists(history_path):
        #         os.makedirs(history_path)
        # except:
        #     logging.error(f"Failed creating directory for prompt history for user: {userId}. Saving failed.")
        #     return

        # curr_date = datetime.today().strftime('%Y_%m_%d')
        # filename = f"{userId}_prompt_history_{curr_date}.txt"
        # file_path = os.path.join(history_path, filename)
        # try:
        #     with open(file_path, "w") as file:
        #         file.write(history_txt)
        # except:
        #     logging.error(f"Failed saving prompt history to file for user: {userId}.")

    def get_usr_prompt(
            userId: str
    ) -> str:
        prompt = None
        try:
            conn = db.connect(database_path)
            cur = conn.cursor()
            update_query = """SELECT prompt FROM usr_prompts WHERE exptUserId = ?"""
            data = [userId]
            cur.execute(update_query, data)
            prompt = cur.fetchone()[0]
        except Error as e:
            logging.error(f"Failed fetching prompt for user: {userId}.")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
        return prompt

    def check_userId(
            userId: str
    ) -> None:
        if int(userId) in userIds:
            print(userId)
        else:
            raise gr.Error(
                f"The following userId is incorrect: ({userId}). Please enter a valid userId and try again."
            )
    
    def check_scenario(
            scenario: str
    ) -> None:
        if scenario in scenarios:
            print(scenario)
        else:
            raise gr.Error(
                f"Please select a scenario and try again."
            )

    def generate(
        message: str,
        history_with_input: list[tuple[str, str]],
        system_prompt: str,
        userId: str,
        scenario: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        request: gr.Request,
    ) -> Iterator[list[tuple[str, str]]]:
        if max_new_tokens > MAX_MAX_NEW_TOKENS:
            raise ValueError
        try:
            new_history = None
            # headers = request.headers
            history = history_with_input[:-1]
            final_system_prompt = get_usr_prompt(userId)
            generator = llama2_wrapper.run(
                message,
                history,
                final_system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            )
            try:
                first_response = next(generator)
                new_history = history + [(message, first_response)]
                yield new_history
            except StopIteration:
                yield history + [(message, "")]
            for response in generator:
                new_history = history + [(message, response)]
                yield new_history
            save_prompt_history(userId, scenario, final_system_prompt, new_history)
        except Exception as e:
            logging.exception(e)

    def check_input_token_length(
        message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> None:
        input_token_length = llama2_wrapper.get_input_token_length(
            message, chat_history, system_prompt
        )
        if input_token_length > MAX_INPUT_TOKEN_LENGTH:
            raise gr.Error(
                f"The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
            )

    max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    temperature = 0.6
    top_p = 0.9
    top_k = 50
    default_advanced_checkbox = False
    database_path = os.path.join(DATA_PATH, DATABASE_NAME)
    userIds = load_userIds()
    scenarios = ["Birthday with Friends/Family", "Long drive", "Un-popular/Niche"]

    # #component-0 #component-1 #component-2 #component-4 #component-5 { height:71vh !important; }
    CSS = """
        .contain { display: flex; flex-direction: column;}
        #component-0 #component-1 #component-24 > div:nth-child(2) { height:80vh !important; overflow-y:auto }
        .text-left-aligned {text-align: left !important; font-size: 16px;}
    """
    with gr.Blocks(css=CSS) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown(DESCRIPTION)
                with gr.Row():
                        userId = gr.Textbox(
                            label="Experiment ID",
                            show_label=True,
                            placeholder="Type your Experiment ID...",
                            max_lines=1,
                        )
                        scenario = gr.Dropdown(
                            label="Scenario",
                            show_label=True,
                            choices=scenarios,
                            value=None,
                        )
                with gr.Group():
                    chatbot = gr.Chatbot(label="MovieLens")
                    with gr.Row():
                        textbox = gr.Textbox(
                            container=False,
                            show_label=False,
                            placeholder="Type a message...",
                            scale=10,
                        )
                        submit_button = gr.Button(
                            "Submit", variant="primary", scale=1, min_width=0
                        )
                with gr.Row():
                    retry_button = gr.Button("🔄  Retry", variant="secondary")
                    undo_button = gr.Button("↩️ Undo", variant="secondary")
                    clear_button = gr.Button("🗑️  Clear", variant="secondary")

                saved_input = gr.State()
                with gr.Column(visible=default_advanced_checkbox) as advanced_column:
                    system_prompt = gr.Textbox(
                        label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=6
                    )
                    max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=1,
                        maximum=MAX_MAX_NEW_TOKENS,
                        step=1,
                        value=DEFAULT_MAX_NEW_TOKENS,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=4.0,
                        step=0.1,
                        value=1.0,
                    )
                    top_p = gr.Slider(
                        label="Top-p (nucleus sampling)",
                        minimum=0.05,
                        maximum=1.0,
                        step=0.05,
                        value=0.95,
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=1,
                        maximum=1000,
                        step=1,
                        value=50,
                    )
        textbox.submit(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        ).then(
            fn=check_userId,
            inputs=[userId],
            api_name=False,
            queue=False,
        ).then(
            fn=check_scenario,
            inputs=[scenario],
            api_name=False,
            queue=False,
        ).success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                userId,
                scenario,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        button_event_preprocess = (
            submit_button.click(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            )
            .then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            )
            .then(
                fn=check_input_token_length,
                inputs=[saved_input, chatbot, system_prompt],
                api_name=False,
                queue=False,
            ).then(
                fn=check_userId,
                inputs=[userId],
                api_name=False,
                queue=False,
            ).then(
                fn=check_scenario,
                inputs=[scenario],
                api_name=False,
                queue=False,
            ).success(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    userId,
                    scenario,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )
        )

        retry_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=check_userId,
            inputs=[userId],
            api_name=False,
            queue=False,
        ).then(
            fn=check_scenario,
            inputs=[scenario],
            api_name=False,
            queue=False,
        ).then(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                userId,
                scenario,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        undo_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=lambda x: x,
            inputs=[saved_input],
            outputs=textbox,
            api_name=False,
            queue=False,
        )

        clear_button.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, saved_input],
            queue=False,
            api_name=False,
        )

    demo.queue(max_size=20).launch(share=args.share) # concurrency_count=5, 


if __name__ == "__main__":
    main()
