import redis
import os
import uuid
import time
import rq


class Redis:
    _redis: redis.Redis = None

    def __new__(cls):
        if cls._redis is not None:
            return cls._redis
        cls.connect()

        return cls._redis

    @classmethod
    def connect(cls):
        cls._redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=os.getenv("REDIS_PORT", 6379),
            decode_responses=False,
        )


class TaskQueue:
    _queue: rq.Queue = None

    def __new__(cls):
        if cls._queue is not None:
            return cls._queue
        cls.connect()

        return cls._queue

    @classmethod
    def connect(cls):
        cls._queue = rq.Queue(connection=Redis())


class Queue:
    queue_key: str

    def __init__(self) -> None:
        self.redis = Redis()
        self.setup_queue()

    def setup_queue(self) -> None:
        self.queue_key = f"queue_{uuid.uuid4().hex}"

    def put(self, data) -> None:
        self.redis.lpush(self.queue_key, data)

    def get(self):
        val = None
        while val is None:
            val = self.redis.rpop(self.queue_key)
        return val
