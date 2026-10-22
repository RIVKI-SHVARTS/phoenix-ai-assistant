"""Local grounded chat plus lexical evidence inspection."""

import os
import hmac
import threading
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, Response
from werkzeug.exceptions import RequestEntityTooLarge
from retrieval import SearchIndex
from answering import AnswerService, AnswerUnavailable, AnswerBudgetExceeded, validate_input

BASE_DIR = Path(__file__).resolve().parent


def create_app(index_root=None, answer_service=None, index_factory=SearchIndex):
    app = Flask(__name__)
    trusted_hosts = ["localhost", "127.0.0.1", "[::1]"]
    render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    if render_host and render_host.endswith('.onrender.com') and '/' not in render_host:
        trusted_hosts.append(render_host)
    app.config.update(MAX_CONTENT_LENGTH=60000, TRUSTED_HOSTS=trusted_hosts)
    root = Path(index_root or os.environ.get("PHOENIX_INDEX_ROOT") or BASE_DIR / "data/index")
    state = {"pointer": None, "index": None, "answer_service": answer_service}
    chat_lock = threading.Lock()

    @app.before_request
    def demo_access():
        password = os.environ.get("DEMO_PASSWORD")
        if not password or request.path == '/healthz':
            return None
        auth = request.authorization
        if (not auth or auth.type != 'basic' or auth.username != 'demo'
                or not hmac.compare_digest((auth.password or '').encode(), password.encode())):
            return Response('Demo access required.', 401,
                            {'WWW-Authenticate': 'Basic realm="Phoenix demo", charset="UTF-8"'})

    def index():
        pointer = (root / "CURRENT.json").read_bytes()
        if state["index"] is None or pointer != state["pointer"]:
            candidate = index_factory(root)
            state.update(pointer=pointer, index=candidate)
        return state["index"]

    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get('/healthz')
    def live():
        return jsonify(status='up')

    @app.get("/api/health")
    def health():
        try:
            current = index()
            return jsonify(
                status="evidence_search",
                index_ready=True,
                chat_ready=(current.manifest["mode"] == "semantic"
                    and current.manifest.get("vector_backend") == "sqlite-numpy-v1"
                    and bool(os.environ.get("OPENAI_API_KEY") or state["answer_service"])),
                search_mode="hybrid" if current.manifest["mode"] == "semantic" else "lexical",
                semantic_ready=current.manifest["mode"] == "semantic",
                pages=current.manifest["included_pages"],
                chunks=len(current.chunks),
            )
        except (OSError, ValueError, KeyError):
            return (
                jsonify(
                    status="index_unavailable", index_ready=False, chat_ready=False
                ),
                503,
            )

    @app.post("/api/search")
    def search():
        if not request.is_json:
            return jsonify(error="יש לשלוח בקשה בפורמט JSON."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"question"}:
            return jsonify(error="נדרש שדה question בלבד."), 400
        question = payload["question"]
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            return jsonify(error="יש להקליד שאלה באורך 1–2000 תווים."), 400
        try:
            result = index().search(question, mode="lexical", limit=6)
            public = []
            for row in result["results"]:
                p = urlsplit(row["page_url"])
                if (
                    p.scheme != "https"
                    or p.hostname != "www.fnx.co.il"
                    or p.username
                    or p.password
                ):
                    continue
                public.append(
                    {
                        key: row[key]
                        for key in (
                            "chunk_id",
                            "page_url",
                            "page_title",
                            "heading",
                            "text",
                            "fetched_at",
                        )
                    }
                )
            return jsonify(
                mode="lexical",
                answer_generated=False,
                results=public,
                notice="אלה קטעים שנמצאו לפי מילים. יש לבדוק שהם עונים לשאלה; זו אינה תשובה של הבוט.",
            )
        except (OSError, ValueError, KeyError):
            return (
                jsonify(
                    error="מאגר החיפוש אינו זמין. יש לבנות אותו לפי הוראות ההפעלה."
                ),
                503,
            )

    @app.post("/api/chat")
    def chat():
        if not request.is_json:
            return jsonify(error="יש לשלוח בקשה בפורמט JSON."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not {"question"} <= set(payload) <= {"question", "history"}:
            return jsonify(error="נדרשים question והיסטוריה אופציונלית בלבד."), 400
        try:
            validate_input(payload["question"], payload.get("history", []))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        if not chat_lock.acquire(blocking=False):
            return jsonify(error="בקשה קודמת עדיין מתבצעת. המתיני לסיומה."), 429
        try:
            current = index()
            if current.manifest["mode"] != "semantic":
                raise AnswerUnavailable("אין אינדקס סמנטי פעיל בתיקייה שממנה הופעל השרת.")
            if state["answer_service"] is None:
                state["answer_service"] = AnswerService(root)
            result = state["answer_service"].answer(current, payload["question"], payload.get("history", []))
            return jsonify(result)
        except AnswerBudgetExceeded as error:
            return jsonify(error=str(error)), 429
        except AnswerUnavailable as error:
            return jsonify(error=str(error)), 503
        except Exception:
            # Do not return raw SDK/database error messages to the browser.
            return jsonify(error="לא ניתן להשלים את המענה כרגע. בדקי שהאינדקס זמין והמפתח מוגדר, ושמרי את הודעת השגיאה לבדיקה."), 503
        finally:
            chat_lock.release()

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error="הבקשה גדולה מדי."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
