import os
from dotenv import load_dotenv

env_path = r"c:\Users\salik\OneDrive\Desktop\GitHub Repo\DocuChat\.env"
print("Env path:", env_path)
print(".env exists:", os.path.exists(env_path))

result = load_dotenv(dotenv_path=env_path, override=True)
print("load_dotenv succeeded:", result)

key = os.environ.get("MISTRAL_API_KEY", "")
print("Key length:", len(key))
print("Key repr (first 10):", repr(key[:10]))
