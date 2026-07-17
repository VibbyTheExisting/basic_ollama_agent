class Callbacks:
    def on_token(self, token: str): pass
    def on_message(self, msg: dict): pass
    def on_messages(self, msgs: list): pass
    def on_tool_call_start(self, name: str, args: list): pass
    def on_tool_call_end(self, name: str, result: str): pass
    def on_tool_approval(self, name: str, args: list): pass
    def on_complete(self): pass
    def on_start(self): pass