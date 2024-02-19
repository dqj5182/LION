import hashlib

def hash_str(file_name):
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    md5 = hashlib.md5()
    data = file_name
    md5.update(data.encode())
    hash_str = md5.hexdigest()[:6]
    return hash_str
