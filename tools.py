import inspect
from ddgs import DDGS
from recording import recorder
from listener import SAMPLE_RATE

### CURRENTLY, THE AGENT DOES NOT HAVE ACCESS TO THIS
def web_search(query: str):
    with DDGS() as ddgs:
        return str(ddgs.text(query, max_results=10))
    
def start_recording():
    recorder.start(SAMPLE_RATE)

def stop_recording():
    recorder.stop()

class tool:
    def __init__(self, fn: callable, description: str, needs_approval = False):
        self.fn, self.description, self.needs_approval = fn, description, needs_approval

    def get_schema(self):
        return {
            "schema": {
                "name": self.fn.__name__,
                "description": self.description,
                "parameters": self.get_params_schema()
            },
            "fn": self.fn,
            "needs_approval": self.needs_approval
        }
    
    def get_params_schema(self):
        sig = inspect.signature(self.fn)
        arg_types = {
            name: (p.annotation.__name__ if hasattr(p.annotation, "__name__") else str(p.annotation))
            for name, p in sig.parameters.items()
        }
        if not arg_types:
            return {}
        required = [
            name for name, p in sig.parameters.items()
            if p.default is inspect._empty
        ]
        return {
            "type": "object",
            "properties": {
                name: {"type": _type} for name, _type in arg_types.items()
            },
            "required": required
        }
    
def build_tools(tools: list[tool]):
    return {
        t.fn.__name__: t.get_schema() for t in tools
    }

raw_tools = [tool(start_recording, "Start a recording of the room.", False), tool(stop_recording, "Stop a recording of the room.", False)]
tools = build_tools(raw_tools)