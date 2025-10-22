from fastapi import FastAPI
from app.core.middleware import setup_cors
from app.routers.base import router as base_router
from app.routers.modelos import router as modelos_router

app = FastAPI()
setup_cors(app)
app.include_router(base_router)
app.include_router(modelos_router)

# solo si se lanza directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
