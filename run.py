from tools import tools
from system_prompt import SYSTEM_PROMPT
import threading
import ollama

DEFAULT_MODEL = "qwen2.5"

def run_agent(
    user_message: str,
    conversation_history: list,
    callbacks,
    tools: dict = tools,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "",
    cancel_event: threading.Event = None
):
    callbacks.on_start()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    callbacks.on_message({"role": "user", "content": user_message})

    if (not model_name):
        model_name = DEFAULT_MODEL

    full_response = ""

    ollama_tools = []
    for _, tool in tools.items():
        ollama_tools.append({
            "type": "function",
            "function": tool["schema"]
        })
    while True:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            tools=ollama_tools,
            stream=True
        )

        tool_calls = []
        current_text = ""
        stream_error = None
        stopped = False

        try:
            for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    callbacks.on_token("_Stopped by user_")
                    current_text += "_Stopped by user_"
                    stopped = True
                    break

                msg = chunk["message"]

                if "content" in msg and msg["content"]:
                    token = msg["content"]
                    current_text += token
                    callbacks.on_token(token)

                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_calls.append(tc)
                        args = tc["function"]["arguments"]
                        callbacks.on_tool_call_start(tc["function"]["name"], args)

        except Exception as e:
            stream_error = e
            if not current_text:
                raise e

        callbacks.on_message({"role": "assistant", "content": current_text})
        full_response += current_text
        if stopped: break

        if stream_error and not current_text:
            full_response = "I couldn't generate a response. Try rephrasing your message."
            break

        if not tool_calls:
            messages.append({"role": "assistant", "content": current_text})
            break

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            args = tc["function"]["arguments"]

            tool_def = tools.get(tool_name)
            if not tool_def:
                continue

            if tool_def.get("needs_approval"):
                approved = callbacks.on_tool_approval(tool_name, args)
                if not approved:
                    return full_response

            try:
                result = tool_def["fn"](**args)
            except Exception as e:
                result = str(e)
            callbacks.on_tool_call_end(tool_name, result)

            messages.append({
                "role": "assistant",
                "tool_calls": [tc]
            })
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result
            })
            callbacks.on_message({
                "role": "assistant",
                "tool_calls": [tc]
            })
            callbacks.on_message({
                "role": "tool",
                "name": tool_name,
                "content": result
            })

    if (cancel_event is None) or (not cancel_event.is_set()):
        callbacks.on_complete()

    return full_response