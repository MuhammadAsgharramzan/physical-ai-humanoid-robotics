from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import FastAPI
import time

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app: FastAPI):
    """Setup rate limiting for the FastAPI application"""
    app.state.limiter = limiter
    app.add_exception_handler(429, _rate_limit_exceeded_handler)

    return limiter