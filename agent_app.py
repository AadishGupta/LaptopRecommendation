# agent_app.py - Laptop Shopping Assistant (LangGraph Multi-Agent edition)
# 
# Architecture: Orchestrator → 4 specialist agent pipelines
#
#   [ORCHESTRATOR]
#       │
#       ├── recommend     → conversation → moderation → intent → search → compare → followup
#       ├── side_compare  → side_compare_parse → side_compare_agent
#       ├── upgrade       → upgrade_node
#       └── pdf_report    → pdf_report_node
#
# Uses Llama 3.1 via Ollama runs on GPU

import os
import sys
import logging
import torch
from flask import Flask, redirect, url_for, render_template, request, send_from_directory, abort
from agent_functions import (
    build_vector_store,
    make_initial_state,
    run_turn,
    LaptopState,
    PDF_OUTPUT_DIR,
    OLLAMA_MODEL,
    CTX_WINDOW,
)
import kg_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================================================================
# GPU OPTIMIZATION CHECKS
# =============================================================================
def check_gpu_availability():
    """Check and report GPU availability."""
    print("\n" + "=" * 60)
    print(" GPU STATUS CHECK")
    print("=" * 60)
    
    # Check PyTorch CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ PyTorch CUDA: Available")
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   Device Count: {gpu_count}")
        
        if gpu_memory < 6:
            print(f"⚠️  Warning: GPU has {gpu_memory:.1f} GB VRAM")
            print(f"   Model requires a few GB VRAM depending on size")
            print(f"   Consider using 4-bit quantization for better performance")
    else:
        print("❌ PyTorch CUDA: Not Available")
        print("   Model will run on CPU (slower)")
        print("   Please install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"✅ Ollama: Available")
            print(f"   Installed models: {', '.join(model_names) if model_names else 'None'}")
            
            # Check if the configured model is installed
            if OLLAMA_MODEL in model_names:
                print(f"✅ {OLLAMA_MODEL}: Installed")
            else:
                print(f"❌ {OLLAMA_MODEL}: Not Installed")
                print(f"   Install with: ollama pull {OLLAMA_MODEL}")
                print(f"   Or use a different model by changing OLLAMA_MODEL in agent_functions.py")
        else:
            print("⚠️  Ollama: Available but API returned unexpected status")
    except Exception as e:
        print(f"⚠️  Ollama: Not Available ({e})")
        print("   Please install and start Ollama:")
        print("   https://ollama.ai/download")
        print(f"   Then run: ollama pull {OLLAMA_MODEL}")
    
    print("=" * 60 + "\n")
    
    # Set environment variables for better GPU utilization
    if cuda_available:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["OLLAMA_GPU"] = "1"
        os.environ["OLLAMA_NUM_GPU"] = "1"
    
    return cuda_available

# =============================================================================
# SESSION STATE  (single user — extend to flask.session / Redis for multi-user)
# =============================================================================
_state: LaptopState = make_initial_state()
_initialized: bool = False

# Cache for model loading status
_gpu_available: bool = False


def ensure_initialized():
    """Initialize the application with GPU optimizations."""
    global _state, _initialized, _gpu_available
    
    if _initialized:
        return
    
    _initialized = True
    _gpu_available = check_gpu_availability()

    print("\n" + "=" * 60)
    print(" 💻 LAPTOP SHOPPING ASSISTANT — LangGraph Multi-Agent")
    print(f" 🤖 Model: {OLLAMA_MODEL}")
    print(" 🔧 GPU: " + ("Enabled" if _gpu_available else "Disabled (CPU)"))
    print(" 📦 Context Window: 4096 tokens")
    print(" Orchestrator + 4 specialist agents")
    print("=" * 60)

    # Ensure PDF output directory exists
    if os.path.exists(PDF_OUTPUT_DIR) and not os.path.isdir(PDF_OUTPUT_DIR):
        os.remove(PDF_OUTPUT_DIR)
    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)

    # Build vector store and knowledge graph. Vector search itself is pure
    # Qdrant (laptops_chunked collection, built by chunk_qdrant_pytorch.py);
    # the laptop catalog used for the feature cache and knowledge graph is
    # also read straight from that same Qdrant collection — no local pkl
    # file is used anywhere in the app.
    try:
        build_vector_store()
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        print("\n⚠️  Error: Could not build vector store.")
        print("   Make sure Qdrant is running and the 'laptops_chunked' collection exists.")
        print("   You may need to run chunk_qdrant_pytorch.py first.\n")
        raise

    # Warm up: send an empty "start" message so the graph emits the welcome turn
    try:
        _state = run_turn(make_initial_state(), "Hello")
        print("✅ Assistant initialized successfully")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        print("\n⚠️  Warning: Initialization failed. Check that Ollama is running.")
        print(f"   Error: {e}\n")

        # Create a minimal state so the app can still start
        _state = make_initial_state()
        conv_bot = [{"bot": f"⚠️  Error initializing assistant: {e}\n\nPlease make sure:\n1. Ollama is running: `ollama serve`\n2. Model is installed: `ollama pull {OLLAMA_MODEL}`\n3. Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`\n\nClick 'New Conversation' to try again."}]
        _state["conversation_bot"] = conv_bot


# =============================================================================
# MAIN ROUTES
# =============================================================================

@app.route("/favicon.ico")
def favicon():
    # Browsers request this automatically on every page load. Without a
    # route it 404'd, hit page_not_found(), which itself crashed trying to
    # render a missing error.html template — turning a harmless favicon
    # request into a logged 500 on every single page load.
    return "", 204


@app.route("/")
def default_func():
    ensure_initialized()
    return render_template("conversation_bot.html", name_xyz=_state["conversation_bot"])


@app.route("/end_conversation", methods=["POST", "GET"])
def end_conv():
    global _state
    ensure_initialized()
    _state = run_turn(make_initial_state(), "Hello")
    return redirect(url_for("default_func"))


@app.route("/conversation", methods=["POST"])
def invite():
    global _state
    ensure_initialized()

    user_input = request.form.get("user_input_message", "").strip()
    logger.info(f"📥 [REQUEST] /conversation user_input={user_input!r}")
    if not user_input:
        return redirect(url_for("default_func"))

    try:
        _state = run_turn(_state, user_input)
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        conv_bot = list(_state.get("conversation_bot", []))
        conv_bot.append({
            "bot": f"❌ I encountered an error processing your request.\n\n**Error:** {e}\n\nPlease try again or start a new conversation."
        })
        _state["conversation_bot"] = conv_bot

    return redirect(url_for("default_func"))


# =============================================================================
# PDF REPORT DOWNLOAD ROUTE
# =============================================================================

@app.route("/static/reports/<path:filename>")
def serve_report(filename: str):
    """Serve generated PDF reports for download."""
    report_dir = os.path.abspath(PDF_OUTPUT_DIR)
    filepath = os.path.join(report_dir, filename)

    # Security: only allow .pdf files inside the reports directory
    if not filename.endswith(".pdf") or not os.path.isfile(filepath):
        abort(404)

    return send_from_directory(
        report_dir,
        filename,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


# =============================================================================
# ADMIN / DEBUG ROUTES
# =============================================================================

@app.route("/admin/status")
def agent_status():
    """Quick health check for the running graph."""
    ensure_initialized()
    return {
        "model": OLLAMA_MODEL,
        "gpu_available": _gpu_available,
        "phase": _state.get("phase"),
        "orchestrator_intent": _state.get("orchestrator_intent"),
        "requirements_complete": _state.get("requirements_complete"),
        "top_3_laptops": bool(_state.get("top_3_laptops")),
        "side_compare_result": bool(_state.get("side_compare_result")),
        "upgrade_advice": bool(_state.get("upgrade_advice")),
        "pdf_url": _state.get("pdf_url", ""),
        "moderation_result": _state.get("moderation_result"),
        "conversation_turns": len(_state.get("conversation_bot", [])),
        "kg_context_count": len(_state.get("kg_context", [])),
        "context_window": CTX_WINDOW,
    }


@app.route("/admin/system")
def system_info():
    """System information for debugging."""
    import psutil
    import torch
    
    info = {
        "model": OLLAMA_MODEL,
        "context_window": CTX_WINDOW,
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        info["gpu_memory_used_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        info["gpu_memory_free_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    
    # System memory
    mem = psutil.virtual_memory()
    info["system_memory"] = {
        "total_gb": round(mem.total / 1024**3, 2),
        "available_gb": round(mem.available / 1024**3, 2),
        "percent_used": mem.percent,
    }
    
    # CPU info
    info["cpu"] = {
        "cores": psutil.cpu_count(),
        "percent": psutil.cpu_percent(interval=0.1),
    }
    
    return info


@app.route("/admin/state")
def dump_state():
    """Return a safe subset of the current state for debugging."""
    ensure_initialized()
    return {
        "phase": _state.get("phase"),
        "orchestrator_intent": _state.get("orchestrator_intent"),
        "requirements": _state.get("requirements"),
        "requirement_string": _state.get("requirement_string"),
        "ranked_count": len(_state.get("ranked_laptops", [])),
        "best_overall": _state.get("best_overall", {}).get("name"),
        "compare_laptops": _state.get("compare_laptops", []),
        "current_laptop": _state.get("current_laptop", ""),
        "pdf_url": _state.get("pdf_url", ""),
        "conversation_turns": len(_state.get("conversation_bot", [])),
    }


@app.route("/admin/clear_cache", methods=["POST"])
def clear_cache():
    """Clear the model cache to free up GPU memory."""
    global _state
    ensure_initialized()
    
    try:
        import gc
        import torch
        
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear Python garbage
        gc.collect()
        
        logger.info("🧹 Cache cleared successfully")
        return {"status": "success", "message": "GPU cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {"status": "error", "message": str(e)}, 500


# =============================================================================
# KG-RAG ADMIN / DEBUG ROUTES
# =============================================================================

@app.route("/admin/kg/stats")
def kg_stats():
    """Graph node/edge counts, node-type breakdown, predicate breakdown."""
    ensure_initialized()
    return kg_rag.graph_stats()


@app.route("/admin/kg/reindex", methods=["POST"])
def kg_reindex():
    """Non-incremental reindex: drop the cache/graph and rebuild from scratch."""
    logger.info("🔄 [ADMIN] /admin/kg/reindex — non-incremental rebuild requested")
    try:
        stats = kg_rag.reindex_non_incremental()
        logger.info("✅ [ADMIN] /admin/kg/reindex — rebuild complete")
        return {"reindexed": True, "stats": kg_rag.graph_stats() if stats is not None else {}}
    except Exception as e:
        logger.error(f"❌ [ADMIN] /admin/kg/reindex — failed: {e}")
        return {"reindexed": False, "error": str(e)}, 500


@app.route("/admin/kg/local_search")
def kg_local_search():
    """Local-level retrieval: single-hop walk seeded at one entity string."""
    ensure_initialized()
    entity = request.args.get("entity", "")
    top_k = int(request.args.get("top_k", 10))
    if not entity:
        abort(400)
    logger.info(f"📍 [ADMIN] /admin/kg/local_search entity='{entity}' top_k={top_k}")
    return kg_rag.local_search(entity, top_k=top_k)


@app.route("/admin/kg/flat_vector_search")
def kg_flat_vector_search():
    """Brute-force nearest-neighbour lookup over the flattened vector subspace."""
    ensure_initialized()
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 10))
    if not query:
        abort(400)
    logger.info(f"🔎 [ADMIN] /admin/kg/flat_vector_search q='{query}' top_k={top_k}")
    return {"results": kg_rag.flat_vector_search(query, top_k=top_k)}


@app.route("/admin/kg/literal_map")
def kg_literal_map():
    """Order-preserving, exact-match token -> node trace for a query string."""
    ensure_initialized()
    text = request.args.get("text", "")
    if not text:
        abort(400)
    logger.info(f"🔗 [ADMIN] /admin/kg/literal_map text='{text}'")
    return {"mapped": kg_rag.literal_sequential_map(text)}


@app.route("/admin/kg/explain/<laptop_id>")
def kg_explain(laptop_id: str):
    """'Why this laptop' — every triplet touching a given laptop id."""
    ensure_initialized()
    logger.info(f"❔ [ADMIN] /admin/kg/explain laptop_id='{laptop_id}'")
    triplets = kg_rag.explain_subgraph(laptop_id)
    return {"laptop_id": laptop_id, "triplets": triplets,
            "context": kg_rag.triplets_to_context(triplets)}


@app.route("/admin/cases/search")
def cases_search():
    """Debug route: retrieve the top-K similar past cases for a query,
    the same lookup compare_node does before generating a recommendation."""
    ensure_initialized()
    from agent_functions import retrieve_similar_cases
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 4))
    pipeline = request.args.get("pipeline") or None
    if not query:
        abort(400)
    logger.info(f"🧠 [ADMIN] /admin/cases/search q='{query}' top_k={top_k} pipeline={pipeline}")
    return {"query": query, "cases": retrieve_similar_cases(query, top_k=top_k, pipeline=pipeline)}


@app.route("/admin/model/status")
def model_status():
    """Check if the configured model is running and ready."""
    ensure_initialized()
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return {
                "ollama_running": True,
                "model_installed": OLLAMA_MODEL in model_names,
                "installed_models": model_names,
                "gpu_enabled": _gpu_available,
            }
        return {"ollama_running": False, "error": "Ollama API returned unexpected status"}
    except Exception as e:
        return {"ollama_running": False, "error": str(e)}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def page_not_found(e):
    # Bug: render_template("error.html", ...) crashed with TemplateNotFound
    # whenever templates/error.html didn't exist — turning EVERY 404 (even a
    # harmless missing /favicon.ico) into an unhandled 500. Try the template
    # first (in case it's added later), fall back to inline HTML so a missing
    # template file can never itself crash the error handler.
    try:
        return render_template("error.html", error="404 - Page Not Found"), 404
    except Exception:
        return "<h1>404 - Page Not Found</h1>", 404


@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {e}")
    try:
        return render_template("error.html", error="500 - Internal Server Error"), 500
    except Exception:
        return "<h1>500 - Internal Server Error</h1>", 500


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 💻 LAPTOP SHOPPING ASSISTANT — LangGraph Multi-Agent")
    print(f" 🤖 Model: {OLLAMA_MODEL}")
    print()
    print("  Orchestrator routes each message to:")
    print("  • Recommendation agent  (default)")
    print("  • Side-by-side Comparison agent")
    print("  • Upgrade Advisor agent")
    print("  • PDF Report agent")
    print()
    print("  Open http://127.0.0.1:5000 in your browser")
    print()
    print("  ADMIN ROUTES:")
    print("  • /admin/status    — check agent status")
    print("  • /admin/system    — system resource usage")
    print("  • /admin/model/status — model status")
    print("  • /admin/clear_cache  — clear GPU cache")
    print("=" * 60 + "\n")

    # Run with production settings
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000,
        threaded=True,  # Handle multiple requests
    )