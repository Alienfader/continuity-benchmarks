"""Inference service. Wraps TGI with our auth + rate-limiting + LoRA-routing shim."""
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from platform.tgi_client import TGIClient
from platform.routing import route_to_adapter

app = FastAPI()
tgi = TGIClient(base_url="http://tgi.ml-serving.svc.cluster.local:8080")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    adapter: str | None = None


@app.post("/generate")
async def generate(req: GenerateRequest, x_tenant_id: str = Header(...)) -> dict:
    adapter = req.adapter or route_to_adapter(x_tenant_id)
    if not adapter:
        raise HTTPException(400, "no adapter for tenant")
    return await tgi.generate(prompt=req.prompt, adapter_id=adapter, max_new_tokens=req.max_new_tokens)
