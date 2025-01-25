import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging import setup_logging
from api.routes import router

app = FastAPI(
    title="Stove Heating AI",
    description="AI service for predicting and optimizing stove heating behavior",
    version="1.0.0",
    root_path="/api/hassio_ingress/{{ADDON_SLUG}}",  # Required for Home Assistant ingress support
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["172.30.32.2"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    setup_logging()

    uvicorn.run(app, host="0.0.0.0", port=8099, reload=False)
