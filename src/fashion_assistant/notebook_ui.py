"""Notebook-native UI. No calls until the shopper clicks a button."""
import json
import uuid
from pathlib import Path

from .agent import ShoppingAgent, nebius_model
from .evaluation import save_run
from .validation import Settings


def make_ui(products, inventory, search, image_root, artifact_root, *, live_enabled=False, model_id=""):
    import ipywidgets as w
    from IPython.display import Image, display
    budget = w.Text(value="2500", description="Budget")
    unit = w.Text(value=products[0]["price_unit"] if products else "CATALOG_UNITS", description="Price unit")
    outfit = w.Dropdown(options=["pair", "set"], description="Outfit")
    top = w.Text(value="M", description="Top/set size")
    bottom = w.Text(value="M", description="Bottom size")
    filters = w.Textarea(value='{}', description="Hard filters", layout=w.Layout(width="95%"))
    mode = w.Dropdown(options=["bm25", "dense", "hybrid", "image", "multimodal"], description="Search")
    query = w.Textarea(value="Suggest a black pure cotton top with blue jeans.", description="Request", layout=w.Layout(width="95%"))
    upload = w.FileUpload(accept="image/*", multiple=False, description="Reference image")
    reset = w.Button(description="Start conversation", button_style="info")
    send = w.Button(description="Send request", button_style="primary")
    output, trace_output = w.Output(), w.Output()
    state = {"agent": None, "signature": None}

    def signature():
        return (budget.value, unit.value, outfit.value, top.value, bottom.value, filters.value, mode.value)

    def start(_):
        with output:
            if not live_enabled:
                print("Live Nebius is disabled. Use the earlier scripted replay for an offline demonstration.")
                return
            try:
                sizes = {"set": top.value} if outfit.value == "set" else {"top": top.value, "bottom": bottom.value}
                settings = Settings(budget.value, sizes, unit.value, json.loads(filters.value))
                state["agent"] = ShoppingAgent(products, inventory, settings, search, nebius_model(model_id), mode=mode.value)
                state["signature"] = signature()
                print("New conversation ready. Each Send request may make up to six paid model calls.")
            except Exception as exc:
                print("Could not start:", type(exc).__name__, str(exc) if isinstance(exc, ValueError) else "Check configuration.")

    def send_request(_):
        send.disabled = True
        send.description = "Working…"
        try:
            with output:
                if state["agent"] is None:
                    print("Start a conversation first.")
                    return
                if state["signature"] != signature():
                    print("Settings changed. Click Start conversation to confirm them.")
                    return
                if not query.value.strip() and not upload.value:
                    print("Enter a request or upload an image.")
                    return
                reference = None
                if upload.value:
                    if mode.value not in {"image", "multimodal"}:
                        print("Choose image or multimodal search to use the uploaded photograph.")
                        return
                    from PIL import Image as PILImage
                    import io
                    entry = upload.value[0]
                    content = bytes(entry["content"])
                    if len(content) > 10*1024*1024:
                        raise ValueError("Use an image smaller than 10 MB")
                    with PILImage.open(io.BytesIO(content)) as image:
                        image.verify()
                    target = Path(artifact_root)/"uploads"
                    target.mkdir(parents=True, exist_ok=True)
                    reference = target/(uuid.uuid4().hex+".image")
                    reference.write_bytes(content)
                print(f"Waiting for {model_id}. This request may involve up to six model calls.", flush=True)
                result = state["agent"].chat(query.value.strip() or "Find clothing similar to my reference image.", reference)
                print(result["status"], "—", result["reply"])
                if result["proposal"]:
                    proposal = result["proposal"]
                    print("Subtotal:", proposal["subtotal"], proposal["price_unit"], "| SIMULATED stock")
                    print("Styling suggestion (subjective):", proposal["style_suggestion"])
                    for item in proposal["items"]:
                        print(item["name"], "|", item["price"], "| size", item["size"])
                        if image_root and item["image_path"]:
                            root = Path(image_root).resolve()
                            image = (root/item["image_path"]).resolve()
                            if image.is_relative_to(root) and image.is_file():
                                display(Image(filename=str(image), width=220))
                path = save_run(result, Path(artifact_root)/"conversations", {"model": model_id, "mode": mode.value})
                print("Trace saved:", path)
                with trace_output:
                    trace_output.clear_output()
                    print(json.dumps(result, indent=2))
        except Exception as exc:
            with output:
                print("Request failed:", type(exc).__name__, "Check the preceding configuration and image/index setup.")
        finally:
            send.disabled = False
            send.description = "Send request"

    reset.on_click(start)
    send.on_click(send_request)
    details = w.Accordion(children=[trace_output])
    details.set_title(0, "Inspect the last tool trace")
    return w.VBox([w.HTML("<b>Fashion assistant</b><br>Historical prices · simulated stock · reference-image retrieval"),
                   budget, unit, outfit, top, bottom,
                   w.HTML('Hard filters example: {"top": {"fabric": "pure cotton"}, "bottom": {"category": "jeans"}}'),
                   filters, mode, reset, query, upload, send, output, details])
