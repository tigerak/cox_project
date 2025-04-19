
from typing import AsyncGenerator

from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
# modules
from config import *
from app.main import router
from main_control import SmartAssistant

assistant = SmartAssistant() 

@router.post("/chat_stream")
async def chat_stream(request: Request):
    """
    POST /api/chat_stream
    body: { "input": "...", "session_id": "..." }
    SSE(Event‑Stream) 로 토큰 실시간 전송
    """
    data = await request.json()
    if not data:
        return JSONResponse({"error": "No data provided"}, status_code=400)

    user_input = data.get('input', '')
    session_id = data.get("session_id", "default") 

    async def token_generator():
        try:
            async for token in assistant.stream_chat(user_input, session_id):
                yield f"{token}".encode("utf-8")
        except Exception as exc:
            # 스트림 중 예외가 발생해도 SSE 형식으로 에러 전송
            yield f"event:error\ndata:{str(exc)}\n\n".encode("utf-8")


    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx proxy 시 버퍼링 방지
        },
    )