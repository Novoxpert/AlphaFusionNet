import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.flushdb()
print("DB 0 cleared!")
r = redis.Redis(host='localhost', port=6379, db=1)
r.flushdb()
print("DB 1 cleared!")
r = redis.Redis(host='localhost', port=6379, db=2)
r.flushdb()
print("DB 2 cleared!")

