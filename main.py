import os
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import requests

# Database helpers
from database import db

app = FastAPI(title="BlaqGPT API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")

# -----------------
# Models
# -----------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = Field(default="gpt-4o-mini")

class ChatResponse(BaseModel):
    reply: str
    usage: Dict[str, Any]

class ImageRequest(BaseModel):
    prompt: str
    size: str = Field(default="1024x1024")

class LimitsResponse(BaseModel):
    month: str
    chat_used: int
    chat_limit: Optional[int]
    image_used: int
    image_limit: Optional[int]
    plan: Literal["free", "basic", "pro"]

# -----------------
# Helpers
# -----------------
FREE_LIMITS = {
    "free": {"chat": 25, "image": 10},
    "basic": {"chat": None, "image": 10},
    "pro": {"chat": None, "image": 25},
}

def current_month_key() -> str:
    now = datetime.utcnow()
    return now.strftime("%Y-%m")

async def get_client_key(request: Request) -> str:
    # Simple anonymous tracking by IP or header; can be replaced with auth later
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"

def get_or_create_usage(client_key: str) -> dict:
    month = current_month_key()
    doc = db["usage"].find_one({"client_key": client_key, "month": month})
    if not doc:
        doc = {
            "client_key": client_key,
            "month": month,
            "chat_used": 0,
            "image_used": 0,
            "plan": "free",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        db["usage"].insert_one(doc)
    return doc

def increment_usage(client_key: str, field: str, inc: int = 1):
    month = current_month_key()
    db["usage"].update_one(
        {"client_key": client_key, "month": month},
        {"$inc": {f"{field}_used": inc}, "$set": {"updated_at": datetime.utcnow()}},
        upsert=True,
    )

# -----------------
# Routes
# -----------------
@app.get("/")
def read_root():
    return {"name": "BlaqGPT API", "message": "Backend running", "version": "0.1.0"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, "name") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response

@app.get("/api/limits", response_model=LimitsResponse)
async def get_limits(request: Request, client_key: str = Depends(get_client_key)):
    usage = get_or_create_usage(client_key)
    plan = usage.get("plan", "free")
    limits = FREE_LIMITS.get(plan, FREE_LIMITS["free"])
    return LimitsResponse(
        month=usage["month"],
        chat_used=usage.get("chat_used", 0),
        chat_limit=limits["chat"],
        image_used=usage.get("image_used", 0),
        image_limit=limits["image"],
        plan=plan,
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, client_key: str = Depends(get_client_key)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    usage = get_or_create_usage(client_key)
    plan = usage.get("plan", "free")
    limits = FREE_LIMITS.get(plan, FREE_LIMITS["free"])
    if limits["chat"] is not None and usage.get("chat_used", 0) >= limits["chat"]:
        raise HTTPException(status_code=402, detail="Chat message limit reached for this month")

    # Call OpenAI chat completions via HTTP
    model = req.model or "gpt-4o-mini"
    payload = {"model": model, "messages": [m.model_dump() for m in req.messages]}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text[:200]}")
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="OpenAI request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

    increment_usage(client_key, "chat", 1)
    new_usage = get_or_create_usage(client_key)
    return ChatResponse(reply=reply, usage={"chat_used": new_usage.get("chat_used", 0)})

@app.post("/api/image")
async def generate_image(req: ImageRequest, request: Request, client_key: str = Depends(get_client_key)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    usage = get_or_create_usage(client_key)
    plan = usage.get("plan", "free")
    limits = FREE_LIMITS.get(plan, FREE_LIMITS["free"])
    if limits["image"] is not None and usage.get("image_used", 0) >= limits["image"]:
        raise HTTPException(status_code=402, detail="Image generation limit reached for this month")

    payload = {
        "model": "gpt-image-1",
        "prompt": req.prompt,
        "size": req.size,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.openai.com/v1/images/generations", json=payload, headers=headers, timeout=60)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text[:200]}")
        data = r.json()
        url = data["data"][0]["url"]
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="OpenAI request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

    increment_usage(client_key, "image", 1)
    return {"url": url, "usage": get_or_create_usage(client_key)}

# Simple admin endpoint to view usage (read-only)
@app.get("/api/admin/usage")
async def admin_usage(key: str):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    month = current_month_key()
    items = list(db["usage"].find({"month": month}, {"_id": 0}).limit(200))
    return {"month": month, "count": len(items), "items": items}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
