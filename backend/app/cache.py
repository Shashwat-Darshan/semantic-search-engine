# backend/app/cache.py
import aioredis
import json
from .config import REDIS_URL

_redis = None

async def get_redis():
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL)
    return _redis

async def get_cached(query):
    r = await get_redis()
    data = await r.get(query)
    if data:
        return json.loads(data)
    return None

async def set_cached(query, result, expire=300):
    r = await get_redis()
    await r.set(query, json.dumps(result), ex=expire)
